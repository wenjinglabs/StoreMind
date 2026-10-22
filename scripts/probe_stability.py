"""重复问同一问题，看答案稳定性。

背景：同一个问题「客人吃到了坏了怎么办」，
第一次答出了「身体不适→建议就医」，第二次没答出。
答案不稳定对演示是致命的——面试官问两次，答案不一样。

这里连跑 N 次，统计：
1. 关键内容是否稳定出现
2. 答案文本有多少种不同版本
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import GraphAgent
from app.agent.llm import build_llm
from app.agent.pipeline import Pipeline
from app.config import settings
from app.rag.embedder import build_embedder
from app.rag.store import build_store
from app.tools.ops_tools import build_ops_tools

N = 5
CASES = [
    ("客人吃到了坏了怎么办", ["身体不适", "就医"]),
    ("满减和折扣券能一起用吗", ["不能叠加"]),
    ("会员卡能跨店用吗", ["跨店"]),
    ("上周广州天河店卖了多少", ["51115"]),
]


def main() -> None:
    settings.ensure_dirs()
    store = build_store(settings)
    emb = build_embedder(settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model)
    llm = build_llm(settings.llm_backend, ollama_base=settings.ollama_base,
                    model=settings.llm_model, base=settings.openai_base,
                    key=settings.openai_key, timeout=settings.llm_timeout)
    tools = build_ops_tools(settings)
    pipe = Pipeline(store, emb, llm, tools, settings)

    out: list[str] = []
    for q, musts in CASES:
        answers = []
        hits = Counter()
        for i in range(N):
            st = pipe.run(q)
            answers.append(st.answer)
            for m in musts:
                if m in st.answer:
                    hits[m] += 1
        uniq = len(set(answers))
        out.append("=" * 72)
        out.append(f"Q: {q}   跑 {N} 次")
        for m in musts:
            rate = hits[m] / N * 100
            flag = "✔ 稳定" if rate == 100 else ("✘ 不稳定" if rate > 0 else "✘ 从未出现")
            out.append(f"   关键内容「{m}」出现 {hits[m]}/{N}  ({rate:.0f}%)  {flag}")
        out.append(f"   答案版本数: {uniq}/{N}")
        for i, a in enumerate(answers, 1):
            out.append(f"   [{i}] {a}")
        out.append("")

    text = "\n".join(out)
    Path("data/stability.txt").write_text(text, encoding="utf-8")
    print("written data/stability.txt")


if __name__ == "__main__":
    main()
