"""索引构建逻辑。抽出来是为了 API 和 CLI 共用同一份实现。

增量策略：按文档内容 hash 判重，未变化直接跳过。
但 hash 变了就整篇重建该文档的 chunk —— 不做 chunk 级别的 diff，
因为改写一个章节后，相邻 chunk 的 overlap 也会变，chunk 级 diff 的收益抵不上复杂度。
"""
from __future__ import annotations

from pathlib import Path

from .rag.pg_store import PgStore
from .config import settings
from .rag.chunker import split_markdown


def ingest_dir(store: PgStore, embedder, corpus_dir: Path) -> dict:
    base = Path(corpus_dir)
    if not base.is_absolute():
        base = settings.corpus_dir.parent / base
    if not base.exists():
        raise FileNotFoundError(f"语料目录不存在: {base}")

    indexed_docs = 0
    indexed_chunks = 0
    skipped = 0
    per_doc: list[dict] = []

    for md in sorted(base.rglob("*.md")):
        raw = md.read_text(encoding="utf-8")
        doc_id = md.relative_to(base).as_posix()
        title = raw.splitlines()[0].lstrip("# ").strip() if raw.strip() else md.stem
        chunks = split_markdown(
            raw, doc_id, title,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        added = store.upsert_doc(doc_id, title, str(md), raw, chunks, embedder)
        if added == 0:
            skipped += 1
        else:
            indexed_docs += 1
            indexed_chunks += added
        per_doc.append({"doc_id": doc_id, "chunks": len(chunks), "reindexed": added > 0})

    return {
        "corpus_dir": str(base),
        "indexed_docs": indexed_docs,
        "indexed_chunks": indexed_chunks,
        "skipped_unchanged": skipped,
        "per_doc": per_doc,
        "stats": store.stats(),
    }
