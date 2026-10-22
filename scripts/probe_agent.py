"""Agent 层实测：自检 + 改写重查到底有没有用。

要回答三个问题：
1. 哪些问题第一轮召回不准、需要重查？
2. 重查之后 top1 有没有变对？
3. 重查让整体指标变好还是变坏？

第 3 条最关键——如果重查只是让 trace 好看但结果没变，那这套东西就是自欺欺人。
"""
from __future__ import annotations

import json
import sys
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

# 刻意挑「用户口语 vs 文档术语」差异大的问题——
# 这类问题第一轮最容易召回不准，是自检该发挥作用的地方。
HARD_CASES = [
    ("客人喝到变质的酒怎么办", "酒水异味"),
    ("酒里有絮状物", "酒水异味"),
    ("经理说要给我打折", "叠加规则"),
    ("两个活动能叠加吗", "叠加规则"),
    ("卡里的钱能在分店刷吗", "跨店使用"),
    ("送的额度能带到别的店吗", "跨店使用"),
    ("不想用了能退钱吗", "退卡与转让"),
    ("把酒寄存在店里行不行", "存酒管理"),
    ("员工能自己给自己打折吗", "内容红线"),
    ("客人拍了视频发到网上", "升级与上报"),
]


def main() -> None:
    settings.ensure_dirs()
    store = build_store(settings)
    emb = build_embedder(settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model)
    llm = build_llm(settings.llm_backend, ollama_base=settings.ollama_base,
                    model=settings.llm_model, base=settings.openai_base,
                    key=settings.openai_key, timeout=settings.llm_timeout)
    tools = build_ops_tools(settings)
    agent = GraphAgent(store, emb, llm, tools, settings)

    out: list[str] = []
    out.append("=" * 78)
    out.append("Agent 自检重查实测")
    out.append("=" * 78)

    n_retry = 0
    n_fixed = 0
    n_broken = 0

    for q, expect in HARD_CASES:
        st = agent.run(q, thread_id="agent-probe")
        retried = any(s.name == "rewrite" for s in st.steps)
        judge_steps = [s for s in st.steps if s.name == "judge"]
        top1 = st.chunks[0].section_path if st.chunks else "-"
        ok = expect in top1
        # 找第一轮召回的 top1（rewrite 之前的 retrieve）
        first_top = "-"
        seen_rewrite = False
        for s in st.steps:
            if s.name == "rewrite":
                seen_rewrite = True
            if s.name == "retrieve" and not seen_rewrite and first_top == "-":
                detail = s.detail
                if "→" in detail:
                    first_top = detail.split("→", 1)[1].strip().split(" | ")[0]

        if retried:
            n_retry += 1
            if ok and expect not in first_top:
                n_fixed += 1
            elif not ok and expect in first_top:
                n_broken += 1

        out.append("")
        out.append(f"Q: {q}")
        out.append(f"   期望章节含「{expect}」 → {'✔' if ok else '✘'}")
        out.append(f"   第一轮 top1: {first_top}")
        out.append(f"   最终   top1: {top1}")
        out.append(f"   重查: {'是' if retried else '否'}")
        for s in judge_steps:
            out.append(f"   judge: {s.detail}")
        for s in st.steps:
            if s.name == "rewrite":
                out.append(f"   rewrite: {s.detail}")
        out.append(f"   答案: {st.answer[:100]}")

    out.append("")
    out.append("=" * 78)
    out.append(f"总用例: {len(HARD_CASES)}")
    out.append(f"触发重查: {n_retry}/{len(HARD_CASES)}")
    out.append(f"  其中重查后修正了 top1: {n_fixed}")
    out.append(f"  其中重查后反而变差:     {n_broken}")
    out.append(f"  最终 top1 命中:         {sum(1 for q, e in HARD_CASES if e in (agent.run(q, thread_id='agent-probe2').chunks[0].section_path if agent.run(q, thread_id='agent-probe2').chunks else ''))}/{len(HARD_CASES)}")

    text = "\n".join(out)
    Path("data/agent_probe.txt").write_text(text, encoding="utf-8")
    agent.close()
    print("written data/agent_probe.txt")


if __name__ == "__main__":
    main()
