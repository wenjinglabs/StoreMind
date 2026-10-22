"""检索诊断 CLI。

用途：当某个问题召回不准时，先确认是向量路丢了还是关键词路丢了。
直接调参是盲调，分开看两路才是定位。

用法:
    python scripts/diagnose.py "满减和折扣券能一起用吗"
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
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.retriever import debug_search  # noqa: E402
from app.rag.store import Store  # noqa: E402

CASES = [
    "满减和折扣券能一起用吗",
    "储值送的金额能不能在别的店花",
    "酒有异味怎么处理",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    settings.ensure_dirs()
    store = Store(settings.db_path)
    embedder = build_embedder(
        settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )

    queries = [args.query] if args.query else CASES
    lines: list[str] = []
    for q in queries:
        d = debug_search(store, embedder, q, k=5)
        lines.append("=" * 70)
        lines.append(f"query    : {d['query']}")
        lines.append(f"expanded : {d['expanded']}")
        for path in ("vector", "bm25", "hybrid"):
            lines.append(f"-- {path} --")
            for i, h in enumerate(d[path], start=1):
                lines.append(f"  {i}. {h['score']:<10} {h['section']}")
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
