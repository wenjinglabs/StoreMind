"""存储与召回：SQLite + 自实现 BM25 + 暴力余弦。

为什么不用向量数据库：这个项目的规模是几百到几万个 chunk，
暴力余弦在这个量级是毫秒级，引入 Qdrant/Milvus 只是多一个要运维的组件。
规模上去之后换 pgvector，SQL 层已经单独放在 deploy/pgvector_init.sql 里。

为什么自己写 BM25 而不是用 FTS5：FTS5 的中文分词依赖 tokenizer 配置，
自己实现字符 bigram 的 BM25，行为完全可解释——面试被问「关键词召回怎么做的」
时，你讲得出每一步。
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunker import Chunk
from .embedder import Embedder, fingerprint

EN_RE = re.compile(r"[a-zA-Z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中文取字符 bigram + 单字，英文数字取整词。"""
    tokens: list[str] = []
    for w in EN_RE.findall(text.lower()):
        tokens.append(w)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    doc_title: str
    section_path: str
    text: str
    score: float
    source: str = ""


SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    doc_title   TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embed_text  TEXT NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id INTEGER NOT NULL,
    term     TEXT NOT NULL,
    tf       INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
);
CREATE INDEX IF NOT EXISTS idx_terms_term ON chunk_terms(term);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---------- 维护 ----------

    def reset(self) -> None:
        """清空所有索引数据但保留库文件。比删库文件稳，也不会被安全策略拦。"""
        with self._conn() as c:
            for t in ("chunk_terms", "chunks", "docs", "meta"):
                c.execute(f"DELETE FROM {t}")

    def get_meta(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def embedder_signature(self) -> str | None:
        """当前索引是用哪个嵌入器建的。

        换嵌入器后端必须重建索引：向量维度会变（hashing 512 维 vs bge-m3 1024 维），
        而 vector_search 靠 `len(v) != len(q)` 跳过维度不符的行——
        也就是说旧向量不会报错，只会被静默丢弃，表现为「召回突然变差」，
        极难排查。所以这里显式记录签名，检测到变化就强制重建。
        """
        return self.get_meta("embedder_sig")

    # ---------- 写入 ----------

    def doc_hash(self, doc_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT content_hash FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return row["content_hash"] if row else None

    def upsert_doc(
        self,
        doc_id: str,
        title: str,
        source_path: str,
        raw: str,
        chunks: list[Chunk],
        embedder: Embedder,
    ) -> int:
        """幂等写入。内容未变则跳过（增量索引的关键）。"""
        h = fingerprint(raw)
        if self.doc_hash(doc_id) == h:
            return 0

        vectors = embedder.embed([ch.embed_text for ch in chunks]) if chunks else []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self._conn() as c:
            old = [r["id"] for r in c.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            if old:
                qs = ",".join("?" * len(old))
                c.execute(f"DELETE FROM chunk_terms WHERE chunk_id IN ({qs})", old)
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute(
                """INSERT INTO docs(doc_id,title,source_path,content_hash,chunk_count,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     title=excluded.title, source_path=excluded.source_path,
                     content_hash=excluded.content_hash, chunk_count=excluded.chunk_count,
                     updated_at=excluded.updated_at""",
                (doc_id, title, source_path, h, len(chunks), now),
            )
            for ch, vec in zip(chunks, vectors):
                cur = c.execute(
                    """INSERT INTO chunks(doc_id,doc_title,section_path,chunk_index,text,embed_text,
                                          char_start,char_end,content_hash,dim,embedding)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        doc_id, ch.doc_title, ch.section_path, ch.chunk_index,
                        ch.text, ch.embed_text, ch.char_start, ch.char_end,
                        fingerprint(ch.text), len(vec), array("f", vec).tobytes(),
                    ),
                )
                cid = cur.lastrowid
                counts: dict[str, int] = {}
                for t in tokenize(ch.embed_text):
                    counts[t] = counts.get(t, 0) + 1
                c.executemany(
                    "INSERT OR REPLACE INTO chunk_terms(chunk_id,term,tf) VALUES(?,?,?)",
                    [(cid, t, n) for t, n in counts.items()],
                )
        return len(chunks)

    def delete_doc(self, doc_id: str) -> None:
        with self._conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            if ids:
                qs = ",".join("?" * len(ids))
                c.execute(f"DELETE FROM chunk_terms WHERE chunk_id IN ({qs})", ids)
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))

    # ---------- 召回 ----------

    def vector_search(self, query_vec: list[float], k: int) -> list[Hit]:
        q = array("f", query_vec)
        if len(q) == 0:
            return []
        hits: list[Hit] = []
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,doc_id,doc_title,section_path,text,embedding FROM chunks"
            ).fetchall()
        for r in rows:
            v = array("f")
            v.frombytes(r["embedding"])
            if len(v) != len(q):
                continue
            # 双方均已 L2 归一化，点积即余弦
            sim = sum(a * b for a, b in zip(q, v))
            hits.append(
                Hit(r["id"], r["doc_id"], r["doc_title"], r["section_path"], r["text"], sim, "vector")
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def bm25_search(self, query: str, k: int, k1: float = 1.5, b: float = 0.75) -> list[Hit]:
        q_terms = [t for t in tokenize(query) if t]
        if not q_terms:
            return []
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            if total == 0:
                return []
            avgdl = c.execute(
                "SELECT AVG(LENGTH(embed_text)) a FROM chunks"
            ).fetchone()["a"] or 1.0
            qs = ",".join("?" * len(q_terms))
            df_rows = c.execute(
                f"SELECT term, COUNT(DISTINCT chunk_id) df FROM chunk_terms WHERE term IN ({qs}) GROUP BY term",
                q_terms,
            ).fetchall()
            df = {r["term"]: r["df"] for r in df_rows}
            if not df:
                return []
            tq = ",".join("?" * len(df))
            rows = c.execute(
                f"""SELECT ct.term, ct.tf, ch.id, ch.doc_id, ch.doc_title, ch.section_path,
                           ch.text, LENGTH(ch.embed_text) dl
                    FROM chunk_terms ct JOIN chunks ch ON ch.id = ct.chunk_id
                    WHERE ct.term IN ({tq})""",
                list(df.keys()),
            ).fetchall()

        scores: dict[int, float] = {}
        payload: dict[int, sqlite3.Row] = {}
        for r in rows:
            term_df = df.get(r["term"], 0)
            idf = math.log(1 + (total - term_df + 0.5) / (term_df + 0.5))
            dl = r["dl"] or 1
            denom = r["tf"] + k1 * (1 - b + b * dl / avgdl)
            gain = idf * (r["tf"] * (k1 + 1)) / denom if denom else 0.0
            scores[r["id"]] = scores.get(r["id"], 0.0) + gain
            payload[r["id"]] = r

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Hit(cid, payload[cid]["doc_id"], payload[cid]["doc_title"],
                payload[cid]["section_path"], payload[cid]["text"], s, "bm25")
            for cid, s in ranked
        ]

    # ---------- 元信息 ----------

    def stats(self) -> dict:
        with self._conn() as c:
            return {
                "docs": c.execute("SELECT COUNT(*) n FROM docs").fetchone()["n"],
                "chunks": c.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"],
                "terms": c.execute("SELECT COUNT(DISTINCT term) n FROM chunk_terms").fetchone()["n"],
                "updated_at": (
                    c.execute("SELECT MAX(updated_at) u FROM docs").fetchone()["u"] or ""
                ),
            }

    def list_docs(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT doc_id,title,chunk_count,updated_at FROM docs ORDER BY doc_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_doc_chunks(self, doc_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT id,section_path,chunk_index,text,char_start,char_end
                   FROM chunks WHERE doc_id=? ORDER BY chunk_index""",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def build_store(settings):
    """按配置选存储后端。

    sqlite 是零依赖/CI 路径；postgres 是部署路径（pgvector + HNSW）。
    两个实现接口一致，上层 Pipeline / Agent / MCP 不需要知道用的是哪个。
    """
    backend = (getattr(settings, "db_backend", "sqlite") or "sqlite").lower()
    if backend == "postgres":
        from .pg_store import PgStore

        return PgStore(settings.pg_dsn, dim=settings.embed_dim)
    return Store(settings.db_path)

