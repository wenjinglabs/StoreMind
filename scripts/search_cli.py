"""检索验证 CLI —— 不接 UI、不接 LLM，只看召回对不对。

这是整个项目最重要的一步：先确认「检索能把对的段落捞出来」，
再谈生成。如果这一步不准，接再强的模型也只是包装错误。

用法:
    python scripts/search_cli.py "会员卡能跨店用吗"
    python scripts/search_cli.py --batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.llm import build_llm  # noqa: E402
from app.agent.pipeline import Pipeline  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.store import Store  # noqa: E402
from app.tools.ops_tools import OpsTools  # noqa: E402

DEMO_QUERIES = [
    "会员卡能跨店用吗",
    "储值送的金额能不能在别的店花",
    "客人说酒有异味怎么办",
    "活动核销失败有哪些原因",
    "满减和折扣券能一起用吗",
    "上周广州天河店卖了多少",
    "广州天河店库存告急的有哪些",
    "新入职第一天要带什么",
]


def _print_state(q: str, st) -> None:
    print("=" * 78)
    print(f"问: {q}")
    print(f"路由: {st.intent.value}  conf={st.confidence:.2f}  ({st.route_reason})")
    print("-" * 78)
    for s in st.steps:
        print(f"  [{s.elapsed_ms:>4}ms] {s.name:<12} {s.detail}")
    print("-" * 78)
    if st.tool_results:
        for t in st.tool_results:
            print(f"工具 {t.tool}: ok={t.ok} note={t.note}")
            print(f"  数据: {t.data}")
    print("召回:")
    for i, h in enumerate(st.chunks, start=1):
        print(f"  [{i}] {h.score:.5f} 《{h.doc_title}》 {h.section_path}  ({h.source})")
    print("-" * 78)
    print("回答:")
    for line in st.answer.splitlines():
        print("  " + line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="单个问题")
    ap.add_argument("--batch", action="store_true", help="跑内置示例问题集")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="", help="输出写入文件（UTF-8）")
    args = ap.parse_args()

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _run(args)

    text = buf.getvalue()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def _run(args) -> None:
    settings.ensure_dirs()
    store = Store(settings.db_path)
    stats = store.stats()
    if stats["chunks"] == 0:
        print("索引为空，先跑 python scripts/build_index.py --rebuild")
        return
    print(f"索引: docs={stats['docs']} chunks={stats['chunks']} terms={stats['terms']}\n")

    embedder = build_embedder(
        settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )
    llm = build_llm(
        settings.llm_backend,
        ollama_base=settings.ollama_base,
        model=settings.llm_model,
        base=settings.openai_base,
        key=settings.openai_key,
    )
    tools = OpsTools(settings.data_dir / "ops.sqlite3")
    pipe = Pipeline(store, embedder, llm, tools, settings)

    queries = DEMO_QUERIES if args.batch else ([args.query] if args.query else DEMO_QUERIES)
    for q in queries:
        _print_state(q, pipe.run(q))


if __name__ == "__main__":
    main()
