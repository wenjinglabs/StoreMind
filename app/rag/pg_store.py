"""PostgreSQL + pgvector 存储实现。

和 `store.py` 的 SQLite 版是同一套接口：`upsert_doc` / `vector_search` /
`bm25_search` / `stats` 等，上面的 Pipeline、Agent、MCP 完全不用改。

两个刻意的决定：

1. **BM25 仍留在应用层**，不换成 PG 的 `tsvector`。
   PostgreSQL 自带全文检索是 TF-IDF 一类，不是 BM25；换掉会改变词法召回，
   现有的 95.8% recall@3 和 12/12 协议冒烟就不可比了。
   `chunk_terms` 表和 SQLite 版同构，打分公式也同一份，迁移前后可对照。

2. **向量列固定维度**。pgvector 的 HNSW 索引要求 `vector(N)`；
   无维度的 `vector` 建 HNSW 会直接报 `column does not have dimensions`。
   所以维度由 `SM_EMBED_DIM` 决定（hashing 512 / bge-m3 1024），
   换嵌入后端时检测到维度变化就重建 chunks 表。
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from .chunker import Chunk
from .embedder import Embedder, fingerprint
from .store import Hit, tokenize


def _vec_literal(vec: list[float]) -> str:
    """把 Python 向量转成 pgvector 的字面量。

    没引 pgvector 的 Python 包：`[...]::vector` 已经够用，
    少一个依赖，也少一层适配。
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class PgStore:
    def __init__(self, dsn: str, dim: int = 512) -> None:
        self.dsn = dsn
        self.dim = int(dim)
        self._local = threading.local()
        self._ensure_schema()

    # ---------- 连接 ----------

    def _conn(self) -> psycopg.Connection:
        """线程内复用连接。

        FastAPI 的线程池会复用线程；每个线程一条 autocommit 连接，
        写操作用显式 transaction 包起来。不引连接池：
        这个规模下多一个依赖不划算，psycopg 的 connect 本身也不慢。
        """
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS docs (
                       doc_id       TEXT PRIMARY KEY,
                       title        TEXT NOT NULL,
                       source_path  TEXT NOT NULL,
                       content_hash TEXT NOT NULL,
                       chunk_count  INTEGER NOT NULL DEFAULT 0,
                       updated_at   TEXT NOT NULL
                   )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS meta (
                       key   TEXT PRIMARY KEY,
                       value TEXT NOT NULL
                   )"""
            )

            # 维度不一致就重建 chunks。HNSW 要求固定维度，不能 ALTER。
            try:
                cur.execute(
                    """SELECT format_type(atttypid, atttypmod) AS t
                       FROM pg_attribute
                       WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"""
                )
                row = cur.fetchone()
            except psycopg.errors.UndefinedTable:
                row = None
            expected = f"vector({self.dim})"
            if row and row["t"] != expected:
                cur.execute("DROP TABLE IF EXISTS chunk_terms")
                cur.execute("DROP TABLE IF EXISTS chunks")

            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS chunks (
                        id           BIGSERIAL PRIMARY KEY,
                        doc_id       TEXT NOT NULL,
                        doc_title    TEXT NOT NULL,
                        section_path TEXT NOT NULL,
                        chunk_index  INTEGER NOT NULL,
                        text         TEXT NOT NULL,
                        embed_text   TEXT NOT NULL,
                        char_start   INTEGER NOT NULL,
                        char_end     INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        dim          INTEGER NOT NULL,
                        embedding    vector({self.dim}) NOT NULL
                    )"""
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            cur.execute(
                """CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                   ON chunks USING hnsw (embedding vector_cosine_ops)
                   WITH (m = 16, ef_construction = 64)"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS chunk_terms (
                       chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                       term     TEXT NOT NULL,
                       tf       INTEGER NOT NULL,
                       PRIMARY KEY (chunk_id, term)
                   )"""
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_terms_term ON chunk_terms(term)")

    # ---------- 维护 ----------

    def reset(self) -> None:
        with self._conn().transaction():
            with self._conn().cursor() as cur:
                cur.execute("TRUNCATE chunk_terms, chunks, docs, meta RESTART IDENTITY")

    def get_meta(self, key: str) -> str | None:
        with self._conn().cursor() as cur:
            cur.execute("SELECT value FROM meta WHERE key=%s", (key,))
            row = cur.fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn().transaction():
            with self._conn().cursor() as cur:
                cur.execute(
                    "INSERT INTO meta(key,value) VALUES(%s,%s) "
                    "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                    (key, value),
                )

    def embedder_signature(self) -> str | None:
        return self.get_meta("embedder_sig")

    # ---------- 写入 ----------

    def doc_hash(self, doc_id: str) -> str | None:
        with self._conn().cursor() as cur:
            cur.execute("SELECT content_hash FROM docs WHERE doc_id=%s", (doc_id,))
            row = cur.fetchone()
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
        h = fingerprint(raw)
        if self.doc_hash(doc_id) == h:
            return 0

        vectors = embedder.embed([ch.embed_text for ch in chunks]) if chunks else []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        conn = self._conn()
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunk_terms WHERE chunk_id IN "
                    "(SELECT id FROM chunks WHERE doc_id=%s)", (doc_id,)
                )
                cur.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))
                cur.execute(
                    """INSERT INTO docs(doc_id,title,source_path,content_hash,chunk_count,updated_at)
                       VALUES(%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (doc_id) DO UPDATE SET
                         title=EXCLUDED.title, source_path=EXCLUDED.source_path,
                         content_hash=EXCLUDED.content_hash, chunk_count=EXCLUDED.chunk_count,
                         updated_at=EXCLUDED.updated_at""",
                    (doc_id, title, source_path, h, len(chunks), now),
                )
                for ch, vec in zip(chunks, vectors):
                    cur.execute(
                        """INSERT INTO chunks(doc_id,doc_title,section_path,chunk_index,text,embed_text,
                                              char_start,char_end,content_hash,dim,embedding)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                           RETURNING id""",
                        (
                            doc_id, ch.doc_title, ch.section_path, ch.chunk_index,
                            ch.text, ch.embed_text, ch.char_start, ch.char_end,
                            fingerprint(ch.text), len(vec), _vec_literal(vec),
                        ),
                    )
                    cid = cur.fetchone()["id"]
                    counts: dict[str, int] = {}
                    for t in tokenize(ch.embed_text):
                        counts[t] = counts.get(t, 0) + 1
                    if counts:
                        cur.executemany(
                            "INSERT INTO chunk_terms(chunk_id,term,tf) VALUES(%s,%s,%s) "
                            "ON CONFLICT DO NOTHING",
                            [(cid, t, n) for t, n in counts.items()],
                        )
        return len(chunks)

    def delete_doc(self, doc_id: str) -> None:
        conn = self._conn()
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunk_terms WHERE chunk_id IN "
                    "(SELECT id FROM chunks WHERE doc_id=%s)", (doc_id,)
                )
                cur.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))
                cur.execute("DELETE FROM docs WHERE doc_id=%s", (doc_id,))

    # ---------- 召回 ----------

    def vector_search(self, query_vec: list[float], k: int) -> list[Hit]:
        if not query_vec:
            return []
        lit = _vec_literal(query_vec)
        with self._conn().cursor() as cur:
            # 45 个 chunk 时是精确结果；语料变大后 HNSW 会接管。
            # ef_search 调高，保证与 SQLite 暴力余弦的对照实验不受近似召回影响。
            cur.execute("SET hnsw.ef_search = 200")
            cur.execute(
                """SELECT id,doc_id,doc_title,section_path,text,
                          1 - (embedding <=> %s::vector) AS score
                   FROM chunks
                   WHERE dim = %s
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (lit, len(query_vec), lit, k),
            )
            rows = cur.fetchall()
        return [
            Hit(r["id"], r["doc_id"], r["doc_title"], r["section_path"],
                r["text"], float(r["score"]), "vector")
            for r in rows
        ]

    def bm25_search(self, query: str, k: int, k1: float = 1.5, b: float = 0.75) -> list[Hit]:
        q_terms = [t for t in tokenize(query) if t]
        if not q_terms:
            return []
        with self._conn().cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM chunks")
            total = cur.fetchone()["n"]
            if total == 0:
                return []
            cur.execute("SELECT AVG(length(embed_text)) AS a FROM chunks")
            avgdl = float(cur.fetchone()["a"] or 1.0)
            cur.execute(
                "SELECT term, COUNT(DISTINCT chunk_id) AS df FROM chunk_terms "
                "WHERE term = ANY(%s) GROUP BY term",
                (q_terms,),
            )
            df = {r["term"]: r["df"] for r in cur.fetchall()}
            if not df:
                return []
            cur.execute(
                """SELECT ct.term, ct.tf, ch.id, ch.doc_id, ch.doc_title, ch.section_path,
                          ch.text, length(ch.embed_text) AS dl
                   FROM chunk_terms ct JOIN chunks ch ON ch.id = ct.chunk_id
                   WHERE ct.term = ANY(%s)""",
                (list(df.keys()),),
            )
            rows = cur.fetchall()

        scores: dict[int, float] = {}
        payload: dict[int, dict] = {}
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
        with self._conn().cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM docs")
            docs = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM chunks")
            chunks = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(DISTINCT term) AS n FROM chunk_terms")
            terms = cur.fetchone()["n"]
            cur.execute("SELECT MAX(updated_at) AS u FROM docs")
            updated = cur.fetchone()["u"] or ""
        return {"docs": docs, "chunks": chunks, "terms": terms, "updated_at": updated}

    def list_docs(self) -> list[dict]:
        with self._conn().cursor() as cur:
            cur.execute(
                "SELECT doc_id,title,chunk_count,updated_at FROM docs ORDER BY doc_id"
            )
            return list(cur.fetchall())

    def get_doc_chunks(self, doc_id: str) -> list[dict]:
        with self._conn().cursor() as cur:
            cur.execute(
                """SELECT id,section_path,chunk_index,text,char_start,char_end
                   FROM chunks WHERE doc_id=%s ORDER BY chunk_index""",
                (doc_id,),
            )
            return list(cur.fetchall())

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            conn.close()
