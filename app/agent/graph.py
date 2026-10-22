"""LangGraph 编排层。

和 pipeline.py 的关系：**同一批节点，两套编排**。

pipeline.py 是手写的顺序调用，graph.py 是 LangGraph 的图。两者共用 Pipeline 的节点方法，
所以 A/B 对比才有意义（scripts/compare_orchestration.py 会验证两条路径输出一致）。

为什么值得换：
1. 并行 fan-out。手写版要自己管「两路都要」时的 gather 和异常合并，
   图里只是一行条件边返回 ["retrieve", "query_data"]，answer 节点自动等两边都完成。
2. checkpoint。手写版要的多轮会话持久化，在这里是 checkpointer 的事，
   thread_id 一给就有，还附带中断恢复能力。
3. 可观测。图上每个节点、每条边都在 trace 里，出事能定位到边而不是整个函数。
"""
from __future__ import annotations

import contextvars
import time
from typing import Annotated, TypedDict

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph

from ..config import Settings
from ..rag.embedder import Embedder
from ..rag.pg_store import PgStore
from ..rag.store import Hit
from ..tools.pg_ops import PgOpsTools
from .judge import EvidenceJudge, merge_hits
from .llm import LLM
from .pipeline import AgentState, Pipeline, Step
from .router import Intent

# trace 用 contextvar 收集，不放进 graph state。
# 原因：state 会被 checkpointer 持久化，trace 一旦进去就会跨轮次累积，
# 第二轮看到第一轮的步骤。trace 是运行时观测数据，不属于业务状态。
_run_trace: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "run_trace", default=None
)


class GraphState(TypedDict, total=False):
    query: str
    search_query: str
    intent: str
    route_reason: str
    confidence: float
    chunks: list
    tool_results: list
    answer: str
    citations: list
    degraded: bool
    turns: list
    # ---- Agent 层：自检与重查 ----
    attempt: int          # 已经重查过几轮
    judge_ok: bool        # 自检结论
    judge_reason: str     # 自检理由，进 trace
    rewritten_query: str  # 改写后的检索词
    max_attempts: int     # 重查上限（跟 Pipeline 共用同一个配置）
    best_chunks: list     # 各轮里得分最高的一轮，最后兜底用
    best_score: float     # best_chunks 的 top1 分数


# 同一会话保留的最大轮数
MAX_TURNS = 6

# 自检不通过时最多重查几轮。
# 设成 2 而不是无限，是因为检索本身有天花板：语料里真没有的内容，
# 改写十次也找不到，反而拖长延迟、让 trace 变得没法看。
MAX_ATTEMPTS = 2


class GraphAgent:
    def __init__(
        self,
        store: PgStore,
        embedder: Embedder,
        llm: LLM,
        tools: PgOpsTools,
        settings: Settings,
    ) -> None:
        self.pipe = Pipeline(store, embedder, llm, tools, settings)
        self.settings = settings
        # Agent 层：证据充分性判定器。规则层先行，模型层兜底。
        self.judge = EvidenceJudge(llm)
        # 会话状态也放 PostgreSQL：运行时不再依赖 SQLite 文件。
        self._conn = psycopg.connect(settings.pg_dsn, autocommit=True)
        # state 里存的是 Hit / ToolResult 这类业务 dataclass。
        # langgraph 的新版序列化默认只信任内置类型，从旧 checkpoint 读回来会警告
        # （未来版本直接拒绝）。这里显式登记，避免服务重启后会话读不回来。
        serde = JsonPlusSerializer(allowed_msgpack_modules=[
            ("app.rag.store", "Hit"),
            ("app.tools.ops_tools", "ToolResult"),
        ])
        self.checkpointer = PostgresSaver(self._conn, serde=serde)
        self.checkpointer.setup()
        self.graph = self._build()

    # ---------- 状态搬运 ----------

    def _to_agent_state(self, d: GraphState) -> AgentState:
        st = AgentState(query=d.get("query", ""))
        st.search_query = d.get("search_query") or st.query
        st.intent = Intent(d.get("intent", "policy"))
        st.route_reason = d.get("route_reason", "")
        st.confidence = float(d.get("confidence", 0.0))
        st.chunks = list(d.get("chunks") or [])
        st.tool_results = list(d.get("tool_results") or [])
        st.judge_ok = bool(d.get("judge_ok", True))
        st.judge_reason = str(d.get("judge_reason") or "")
        st.attempt = int(d.get("attempt", 0))
        return st

    @staticmethod
    def _steps_of(st: AgentState, from_index: int) -> list[dict]:
        return [
            {"step": s.name, "detail": s.detail, "elapsed_ms": s.elapsed_ms}
            for s in st.steps[from_index:]
        ]

    @staticmethod
    def _emit(steps: list[dict]) -> dict:
        buf = _run_trace.get()
        if buf is not None:
            buf.extend(steps)
        return {}

    # ---------- 节点 ----------

    def _route(self, d: GraphState) -> dict:
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_route(st)
        self._emit(self._steps_of(st, n0))
        return {
            "intent": st.intent.value,
            "route_reason": st.route_reason,
            "confidence": st.confidence,
        }

    def _retrieve(self, d: GraphState) -> dict:
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_retrieve(st)
        self._emit(self._steps_of(st, n0))
        return {"chunks": st.chunks}

    def _query_data(self, d: GraphState) -> dict:
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_query_data(st)
        self._emit(self._steps_of(st, n0))
        return {"tool_results": st.tool_results}

    def _answer(self, d: GraphState) -> dict:
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_answer(st)
        self._emit(self._steps_of(st, n0))
        return {
            "answer": st.answer,
            "citations": st.citations,
            "degraded": st.degraded,
        }

    # ---------- Agent 层：自检与改写重查 ----------

    def _judge(self, d: GraphState) -> dict:
        """自检：手上的证据够不够回答这个问题。

        这是「流水线」和「Agent」的分界线。流水线到这里就直接生成答案了；
        Agent 会先问一句「够吗」，不够就改写检索词重来。

        判定本身复用 Pipeline 的判定器（同一个 EvidenceJudge 实例），
        所以手写版和图版对「够不够」的判断必然一致。
        """
        st = self._to_agent_state(d)
        t0 = time.perf_counter()
        has_data = any(getattr(t, "ok", False) for t in st.tool_results)
        res = self.judge.judge(d.get("search_query") or st.query, st.chunks, has_data)

        # 记录本轮：分数更高就更新 best（跟手写版的 rounds 列表等价）
        cur_score = st.chunks[0].score if st.chunks else 0.0
        best_score = float(d.get("best_score", 0.0))
        best_chunks = list(d.get("best_chunks") or [])
        if cur_score > best_score:
            best_score = cur_score
            best_chunks = list(st.chunks)

        detail = res.reason
        if res.rewritten_query:
            detail += f"；改写为「{res.rewritten_query}」"
        self._emit([{"step": "judge", "detail": detail, "elapsed_ms": int((time.perf_counter() - t0) * 1000)}])

        update: dict = {
            "judge_ok": res.sufficient,
            "judge_reason": res.reason,
            "rewritten_query": res.rewritten_query,
            "best_score": best_score,
            "best_chunks": best_chunks,
        }
        # 各轮都没判充足 → 用得分最高的那一轮，保证不差于第一轮
        if (not res.sufficient
                and int(d.get("attempt", 0)) >= int(d.get("max_attempts", MAX_ATTEMPTS))
                and best_chunks):
            update["chunks"] = best_chunks
            self._emit([{"step": "judge", "detail": f"各轮均未判定充足，采用得分最高的一轮（{best_score:.5f}）", "elapsed_ms": 0}])
        return update

    def _rewrite(self, d: GraphState) -> dict:
        """把改写后的检索词写进 search_query，并累加重试次数。

        为什么不直接改 query：query 是用户原话，要保留在答案和 trace 里；
        search_query 才是拿去检索的。两者分开是「那退卡呢」实验的结论。
        """
        attempt = int(d.get("attempt", 0)) + 1
        original = d.get("search_query") or d.get("query", "")
        # 用 compose_query 拼接而不是替换，跟手写版完全一致
        new_sq = EvidenceJudge.compose_query(original, d.get("rewritten_query") or "")
        self._emit([{"step": "rewrite", "detail": f"第 {attempt} 次重查，检索词：{new_sq}", "elapsed_ms": 0}])
        return {"attempt": attempt, "search_query": new_sq}

    @staticmethod
    def _branch_after_judge(d: GraphState) -> str:
        """自检通过 → 生成；不通过且还有次数 → 改写重查。

        这条条件边是整个项目里唯一的「运行时决策」——
        走哪条路取决于模型对证据质量的判断，不是代码写死的。
        """
        if d.get("judge_ok", True):
            return "answer"
        if int(d.get("attempt", 0)) >= int(d.get("max_attempts", MAX_ATTEMPTS)):
            return "answer"
        return "rewrite"

    def _retrieve_again(self, d: GraphState) -> dict:
        """自检不通过后的重查。

        为什么单独一个节点，而不是循环回 retrieve_guard：
        retrieve_guard 是 join 的成员（`add_edge(["retrieve_guard","data_guard"], "merge")`），
        循环回它会破坏 join 语义——join 等的是「本次 superstep 里两个成员都到过」，
        只回一个成员的话 merge 永远等不到另一个。
        这是本项目踩过的第三个 LangGraph 坑，写在这里备忘。

        片段是**替换**而不是合并：每一轮独立判定，
        最终喂给模型的是一组自洽的片段。
        早先版本把几轮结果取并集，模型看到的是一锅粥——
        实测 top1 修对了但答案反而变差。
        """
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_retrieve(st)
        self._emit(self._steps_of(st, n0))
        return {"chunks": st.chunks}

    # ---------- 图 ----------

    def _retrieve_guard(self, d: GraphState) -> dict:
        """纯数据意图不检索，但仍然走这个节点。

        为什么需要 guard：LangGraph 的 join（add_edge(["a","b"], "c")）要求列表里
        每个节点**都完成过**。如果 policy 意图下跳过 retrieve，join 就永远等不到，
        answer 不执行——实测答案是空字符串。
        所以两个分支都必然执行，不需要的那条直接返回空 dict。
        """
        if d.get("intent") == "data":
            self._emit([{"step": "retrieve", "detail": "数据意图，跳过知识库", "elapsed_ms": 0}])
            return {}
        return self._retrieve(d)

    def _data_guard(self, d: GraphState) -> dict:
        if d.get("intent") == "policy":
            self._emit([{"step": "query_data", "detail": "规则意图，跳过结构化查询", "elapsed_ms": 0}])
            return {}
        return self._query_data(d)

    def _merge(self, d: GraphState) -> dict:
        """汇合点：两路都跑完后决定要不要补一次知识库检索。

        逻辑复用 Pipeline._node_merge_fallback，手写版和图版不会跑偏。
        """
        st = self._to_agent_state(d)
        n0 = len(st.steps)
        self.pipe._node_merge_fallback(st)
        extra = self._steps_of(st, n0)
        self._emit(extra)
        return {"chunks": st.chunks}

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("route", self._route)
        g.add_node("retrieve_guard", self._retrieve_guard)
        g.add_node("data_guard", self._data_guard)
        g.add_node("merge", self._merge)
        g.add_node("judge", self._judge)
        g.add_node("rewrite", self._rewrite)
        g.add_node("retrieve_again", self._retrieve_again)
        g.add_node("answer", self._answer)

        g.set_entry_point("route")
        g.add_edge("route", "retrieve_guard")
        g.add_edge("route", "data_guard")
        g.add_edge(["retrieve_guard", "data_guard"], "merge")

        # Agent 层：自检 → 通过则成文，不通过则改写重查。
        g.add_edge("merge", "judge")
        g.add_conditional_edges(
            "judge",
            self._branch_after_judge,
            {"answer": "answer", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve_again")
        # 重查完再自检一次 —— 这条边构成了图里唯一的环。
        g.add_edge("retrieve_again", "judge")

        return g.compile(checkpointer=self.checkpointer)

    # ---------- 对外 ----------

    @staticmethod
    def _build_search_query(query: str, turns: list[str]) -> str:
        """指代消解的最小实现：默认不拼接，只在当前 query 没有实词时才借上文。

        第一版是无条件把上一轮问题拼在前面，实测是有害的：
        「那退卡呢」拼成「会员卡能跨店用吗 那退卡呢」之后，
        「跨店」的权重压过了「退卡」，top1 反而还是「跨店使用」——
        而单用「那退卡呢」去检索，「退卡与转让」就是 top1。
        结论：上下文只在信息不足时补，不要默认补。
        """
        if not turns:
            return query
        stripped = query
        for w in ("那", "这个", "那个", "它", "还有", "呢", "的话", "?", "？", "，"):
            stripped = stripped.replace(w, "")
        if len(stripped.strip()) >= 2:
            return query
        return f"{turns[-1]} {query}"

    def run(self, query: str, thread_id: str = "default") -> AgentState:
        prev = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        prev_vals = dict(prev.values) if prev and prev.values else {}
        turns = list(prev_vals.get("turns") or [])[-MAX_TURNS:]

        init: GraphState = {
            "query": query,
            "search_query": self._build_search_query(query, turns),
            "turns": turns + [query],
            "chunks": [],
            "tool_results": [],
            "attempt": 0,
            "judge_ok": True,
            "max_attempts": int(getattr(self.settings, "agent_max_attempts", MAX_ATTEMPTS)),
        }

        buf: list[dict] = []
        token = _run_trace.set(buf)
        try:
            out = self.graph.invoke(init, config={"configurable": {"thread_id": thread_id}})
        finally:
            _run_trace.reset(token)

        st = self._to_agent_state(out)  # type: ignore[arg-type]
        st.answer = out.get("answer", "")
        st.citations = list(out.get("citations") or [])
        st.degraded = bool(out.get("degraded"))
        st.steps = [Step(s["step"], s["detail"], s["elapsed_ms"]) for s in buf]
        return st

    def history(self, thread_id: str = "default") -> list[str]:
        prev = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list((prev.values or {}).get("turns") or []) if prev else []

    def reset(self, thread_id: str = "default") -> None:
        try:
            self.checkpointer.delete_thread(thread_id)
        except Exception:
            pass

    def close(self) -> None:
        self._conn.close()
