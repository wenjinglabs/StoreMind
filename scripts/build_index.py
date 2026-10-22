"""建索引 CLI。

用法:
    python scripts/build_index.py
    python scripts/build_index.py --corpus corpus --rebuild
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ingest import ingest_dir  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.store import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--rebuild", action="store_true", help="先清空索引再重建")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--out", default="", help="把输出写入该文件（UTF-8），解决 Windows 管道编码问题")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    settings.ensure_dirs()
    store = Store(settings.db_path)
    if args.rebuild:
        store.reset()
        store = Store(settings.db_path)

    embedder = build_embedder(
        settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )

    # 嵌入器换过就强制重建。
    # 不这么做的话：hashing(512维) 切到 bge-m3(1024维)，旧向量会被
    # vector_search 里的 `len(v) != len(q)` 静默跳过，
    # 表现为「召回率突然掉了」而不是「配置不一致」，几乎无法定位。
    prev_sig = store.embedder_signature()
    cur_sig = embedder.signature()
    force_rebuild = False
    if prev_sig and prev_sig != cur_sig and not args.rebuild:
        force_rebuild = True
        store.reset()
        store = Store(settings.db_path)

    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = ROOT / corpus

    result = ingest_dir(store, embedder, corpus)
    store.set_meta("embedder_sig", cur_sig)

    if args.json:
        emit(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        emit(f"corpus      : {result['corpus_dir']}")
        emit(f"embedder    : {embedder.name} (dim={embedder.dim})")
        if force_rebuild:
            emit(f"!! 嵌入器已变更 ({prev_sig} -> {cur_sig})，已强制重建索引")
        emit(f"chunk 参数  : {settings.chunk_min_chars}~{settings.chunk_max_chars} 字, "
             f"overlap {settings.chunk_overlap_chars}")
        emit(f"新索引文档  : {result['indexed_docs']}")
        emit(f"新索引片段  : {result['indexed_chunks']}")
        emit(f"跳过未变更  : {result['skipped_unchanged']}")
        st = result["stats"]
        emit(f"索引总量    : docs={st['docs']} chunks={st['chunks']} terms={st['terms']}")
        emit("")
        emit("每篇文档:")
        for d in result["per_doc"]:
            flag = "重建" if d["reindexed"] else "跳过"
            emit(f"  [{flag}] {d['doc_id']:<40} chunks={d['chunks']}")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
