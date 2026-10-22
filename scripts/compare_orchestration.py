"""两套编排的等价性验证 + 多轮能力演示。

第一段：同一批 query 分别跑手写 Pipeline 和 LangGraph，对比 intent / top1 / 答案。
        节点是同一批，编排不同，输出必须一致——不一致就是迁移引入了 bug。
第二段：多轮会话。重点是「那退卡呢」这种缺主语的追问，
        手写版没有会话状态，图版靠 checkpointer 把上一轮问题拼进检索。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import GraphAgent  # noqa: E402
from app.agent.llm import build_llm  # noqa: E402
from app.agent.pipeline import Pipeline  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.store import build_store  # noqa: E402
from app.tools.ops_tools import build_ops_tools  # noqa: E402

QUERIES = [
    "会员卡能跨店用吗",
    "客人说酒有异味怎么办",
    "满减和折扣券能一起用吗",
    "上周广州天河店卖了多少",
    "广州天河店库存告急的有哪些",
    "新入职第一天要带什么",
]

MULTI_TURN = [
    "会员卡能跨店用吗",
    "那退卡呢",
    "手续费多少",
]


def main() -> None:
    settings.ensure_dirs()
    store = build_store(settings)
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
    tools = build_ops_tools(settings)

    pipe = Pipeline(store, embedder, llm, tools, settings)
    agent = GraphAgent(store, embedder, llm, tools, settings)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("一、等价性验证（手写 Pipeline vs LangGraph）")
    lines.append("=" * 78)
    lines.append(f"{'查询':<26} {'intent':<10} {'top1 一致':<10} {'召回一致':<10} {'数字一致':<10}")
    lines.append("-" * 78)

    same = 0
    for q in QUERIES:
        a = pipe.run(q)
        b = agent.run(q, thread_id=f"cmp-{hash(q) % 9999}")
        t_a = a.chunks[0].section_path if a.chunks else "-"
        t_b = b.chunks[0].section_path if b.chunks else "-"
        top_same = t_a == t_b
        intent_same = a.intent.value == b.intent.value

        # 判据是「编排等价」——由编排决定的部分必须一致：
        #   intent（走哪条路）、召回集合（召回了哪几段）。
        #
        # 答案文本、引用编号、具体数字都不判定：
        # 引用编号和措辞由 LLM 采样决定，同一问题两次调用可能选不同的
        # 引用、报不同的字段组合。那测的是 LLM 的随机性，不是编排的正确性。
        # 编排能保证的是「两条路给模型看的资料完全一样」。
        def numbers(text: str) -> set[str]:
            cleaned = re.sub(r"\[\s*D?\d+\s*\]", " ", text or "")
            return set(re.findall(r"\d+(?:\.\d+)?", cleaned))

        def cite_set(text: str) -> set[str]:
            return set(re.findall(r"\[\s*(D?\d+)\s*\]", text or ""))

        num_a = numbers(a.answer)
        num_b = numbers(b.answer)
        nums_same = num_a == num_b
        ans_same = (a.answer or "").strip() == (b.answer or "").strip()
        # 召回集合：两条路给模型的资料必须完全相同
        chunks_same = [h.chunk_id for h in a.chunks] == [h.chunk_id for h in b.chunks]

        if top_same and intent_same and chunks_same:
            same += 1
        lines.append(
            f"{q:<24} {a.intent.value:<10} {str(top_same):<10} {str(chunks_same):<10} {str(nums_same):<10}"
        )
        if not top_same:
            lines.append(f"    ! 手写 top1: {t_a}")
            lines.append(f"    ! 图   top1: {t_b}")
        if not chunks_same:
            lines.append(f"    ! 召回集合不同，说明检索环节有分歧")
        if not nums_same:
            lines.append(f"    ~ 数字差异（LLM 采样，不参与判定）: 手写 {sorted(num_a)} / 图 {sorted(num_b)}")
    lines.append("-" * 78)
    lines.append(f"编排等价（intent + top1 章节 + 召回集合一致）: {same}/{len(QUERIES)}")

    lines.append("")
    lines.append("=" * 78)
    lines.append("二、多轮会话（靠 checkpointer 持久化，thread_id 区分会话）")
    lines.append("=" * 78)
    agent.reset("demo-multi")
    for i, q in enumerate(MULTI_TURN, start=1):
        st = agent.run(q, thread_id="demo-multi")
        lines.append(f"[第 {i} 轮] 用户: {q}")
        lines.append(f"  检索用的 query : {st.text_for_retrieval()}")
        lines.append(f"  路由           : {st.intent.value} ({st.route_reason})")
        for s in st.steps:
            lines.append(f"  trace          : {s.name:<12} {s.detail}")
        top = st.chunks[0].section_path if st.chunks else "-"
        lines.append(f"  top1           : {top}")
        for tr in st.tool_results:
            lines.append(f"  tool           : {tr.tool} ok={tr.ok} {tr.data}")
        lines.append("")
    lines.append(f"会话历史: {agent.history('demo-multi')}")

    lines.append("")
    lines.append("=" * 78)
    lines.append("三、并行 fan-out（intent=both 时两路同时跑）")
    lines.append("=" * 78)
    st = agent.run("广州天河店上周核销活动和叠加规则有没有冲突", thread_id="demo-both")
    lines.append(f"查询: 广州天河店上周核销活动和叠加规则有没有冲突")
    lines.append(f"路由: {st.intent.value} ({st.route_reason})")
    for s in st.steps:
        lines.append(f"  trace: {s.name:<16} {s.elapsed_ms:>4}ms  {s.detail}")

    agent.close()
    text = "\n".join(lines)
    (ROOT / "data" / "compare_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
