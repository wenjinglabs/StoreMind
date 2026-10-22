"""Agent 工作流：路由 → 检索/查询 → 生成。

这是一个显式状态机，每个节点独立可测、每步都有 trace。
之所以先手写而不直接上 LangGraph：状态机的形状要先自己画一遍，
才知道哪些边该是条件边、哪些状态需要持久化。手写通了，换 LangGraph 是平移。

trace 是刻意设计的：每一步都记下来，演示时直接摊开给面试官看
「模型看到的是什么、检索召回了什么、为什么走了这条分支」。
没有 trace 的 agent 是不可调试的。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..tools.ops_tools import ToolResult
from ..tools.pg_ops import PgOpsTools
from ..rag.retriever import hybrid_search
from ..rag.store import Hit
from ..rag.pg_store import PgStore
from .judge import EvidenceJudge, merge_hits
from .llm import LLM
from .router import Intent, classify

# 自检不通过时最多重查几轮。
# 设成 2 而不是无限，是因为检索有天花板：语料里真没有的内容，
# 改写十次也找不到，反而拖长延迟、让 trace 变得没法看。
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """你是连锁酒吧的门店运营助手，服务对象是门店店员和店长。

规则：
1. 只根据提供的「参考资料」和「经营数据」回答，不要编造。
2. 判定有无答案时，要同时看两个来源：「参考资料」为空但「经营数据」有内容，
   就算有答案，必须用经营数据回答。只有「参考资料」和「经营数据」都没有内容，
   才说「现有资料里没有相关规定」，并建议联系谁。
   经营数据列了几条，就答几条，不要只挑其中一条。
   一条数据里有几个字段，就把几个字段都答出来，不要只报其中一个。
3. 涉及金额、期限、比例、数量时，必须原样引用资料中的数字。
4. 回答末尾用 [1] [2] 标注引用来源编号。编号只在「参考资料」里出现，
   经营数据用 [D1] 这种标号，不要给数据编 [1] [2]。
   纯数据问题（参考资料为空）不需要任何引用标记。
5. 语气简洁，像老员工带新人，不要客套话。

输出格式（必须严格遵守）：
- 只输出最终答案正文，一到三句话。写完就停。
- 不要复述上面的规则，不要复述问题，不要写「首先」「我需要」「根据参考资料」这类开场白。
- 不要输出思考过程、推理步骤或自我检查。
- 不要用列表、标题或加粗，只写连贯的段落。

示例（合格）：
满减券和折扣券不能叠加，同一订单只能选其一，需要服务员在结账前口头确认。[1]

示例（不合格）：
首先，用户问的是满减和折扣券能否一起使用。根据资料[1]，规则明确说明不能叠加……
"""

STORE_RE = re.compile(r"([\u4e00-\u9fff]{2,6}店)")  # 已废弃，改用门店白名单，保留以免破坏外部引用
PERIOD_MAP = (
    (("今天", "今日"), "today"),
    (("昨天", "昨日"), "yesterday"),
    (("上周", "最近七天", "近七天", "这周", "本周"), "last_7d"),
    (("近三十天", "最近三十天", "上月", "上个月"), "last_30d"),
    (("本月", "这个月", "当月"), "this_month"),
)


@dataclass
class Step:
    name: str
    detail: str
    elapsed_ms: int


@dataclass
class AgentState:
    query: str
    # 多轮场景下实际用于检索的 query。指代消解后可能比原始 query 长。
    search_query: str = ""
    intent: Intent = Intent.POLICY
    route_reason: str = ""
    confidence: float = 0.0
    chunks: list[Hit] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    degraded: bool = False
    # ---- Agent 层：自检与重查 ----
    judge_ok: bool = True      # 自检结论：证据够不够
    judge_reason: str = ""     # 自检理由
    attempt: int = 0           # 已重查几轮

    def text_for_retrieval(self) -> str:
        return self.search_query or self.query

    def trace(self) -> list[dict]:
        return [{"step": s.name, "detail": s.detail, "elapsed_ms": s.elapsed_ms} for s in self.steps]


def _extract_period(q: str) -> str:
    for words, period in PERIOD_MAP:
        if any(w in q for w in words):
            return period
    return "last_7d"


class Pipeline:
    def __init__(self, store: PgStore, embedder, llm: LLM, tools: PgOpsTools, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.tools = tools
        self.s = settings
        # Agent 层：证据充分性判定器。手写版是 while 循环，图版是环，
        # 两边共用同一个判定器和同一套合并规则，等价性测试才有意义。
        self.judge = EvidenceJudge(llm)
        self.judge.load_sections(store)

    # ---------- 单个节点 ----------

    def _node_route(self, state: AgentState) -> None:
        t0 = time.perf_counter()
        r = classify(state.query)
        state.intent = r.intent
        state.route_reason = r.reason
        state.confidence = r.confidence
        state.steps.append(
            Step("route", f"intent={r.intent.value} conf={r.confidence:.2f} ({r.reason})",
                 int((time.perf_counter() - t0) * 1000))
        )

    def _node_retrieve(self, state: AgentState) -> None:
        t0 = time.perf_counter()
        state.chunks = hybrid_search(
            self.store, self.embedder, state.text_for_retrieval(),
            top_k=self.s.final_top_k, vector_k=self.s.vector_top_k,
            bm25_k=self.s.bm25_top_k, rrf_k=self.s.rrf_k,
        )
        # 拒答阈值：留了口子但默认不启用，这是刻意的。
        # 标定结果是不可用——命中组最低 BM25 11.37、无答案组最高 11.67，区间重叠。
        # RRF 融合分更是天生不可解释（永远都是 1/(60+rank) 那个量级）。
        # 结论：检索分数不能判断「资料够不够回答」，那是语义判断，应该交给生成模型。
        # 所以 abstention 放在 prompt 层做（见 SYSTEM_PROMPT 第 2 条），不在这里硬凑。
        if self.s.min_relevance > 0:
            kept = [h for h in state.chunks if h.score >= self.s.min_relevance]
            if len(kept) != len(state.chunks):
                state.steps.append(
                    Step("abstain", f"低于阈值 {self.s.min_relevance} 丢弃 {len(state.chunks)-len(kept)} 段", 0)
                )
            state.chunks = kept
        preview = " | ".join(f"{h.section_path}" for h in state.chunks[:3]) or "无召回"
        state.steps.append(
            Step("retrieve", f"召回 {len(state.chunks)} 段 → {preview}",
                 int((time.perf_counter() - t0) * 1000))
        )

    def _node_merge_fallback(self, state: AgentState) -> None:
        """两路都跑完后，结构化查询没结果就回落知识库检索。

        图版里这是 merge 节点，手写版里是这一段。抽出来是为了让两边共用。
        """
        results = state.tool_results
        if any(getattr(t, "ok", False) for t in results):
            state.steps.append(Step("merge", "工具已返回数据，不回落", 0))
            return
        if state.chunks:
            state.steps.append(Step("merge", "已有知识库召回，不回落", 0))
            return
        if state.intent != Intent.DATA:
            state.steps.append(Step("merge", "两路均无结果", 0))
            return
        self._node_retrieve(state)
        state.steps.append(Step("merge", "结构化查询无结果，回落知识库检索", 0))

    def _node_judge_retry(self, state: AgentState) -> None:
        """自检 → 不够就改写重查。

        手写版的实现是 while 循环；图版里同样的逻辑展开成
        judge → rewrite → retrieve_again → judge 这个环。
        两边的判定器、合并规则、重试上限都相同。

        关键设计：**每一轮独立判定，最终只用某一轮的片段，不做并集。**

        早先的版本是把几轮召回合并成一个列表（取并集），
        好处是「重查不会让结果变差」，坏处更严重：
        三轮召回混在一起，模型看到的是一锅粥。
        实测「客人边吃饭边做爱怎么办」这条，top1 修对了（升级与上报），
        但答案却变成「现有资料里没有相关规定」——比没有 Agent 层时更差。
        **检索指标涨了，答案质量反而跌了。**

        改成每轮独立判定后：
        - 某一轮被判 OK，就用那一轮的片段
        - 全都不 OK，用分数最高的那一轮
        这样最终喂给模型的永远是一组自洽的片段。

        返回是否发生过重查。
        """
        has_data = any(getattr(t, "ok", False) for t in state.tool_results)
        max_attempts = getattr(self.s, "agent_max_attempts", MAX_ATTEMPTS)

        # 记录每一轮的片段和它的得分，最后挑最好的
        rounds: list[tuple[float, list[Hit]]] = []

        for attempt in range(max_attempts + 1):
            rounds.append((self._top_score(state.chunks), list(state.chunks)))
            res = self.judge.judge(state.text_for_retrieval(), state.chunks, has_data)
            detail = res.reason
            if res.rewritten_query:
                detail += f"；改写为「{res.rewritten_query}」"
            state.steps.append(Step("judge", detail, 0))
            state.judge_ok = res.sufficient
            state.judge_reason = res.reason

            if res.sufficient:
                return attempt > 0
            if attempt >= max_attempts:
                break

            # 不够 → 改写检索词，重查（这一轮的片段会被下一轮替换）
            state.search_query = EvidenceJudge.compose_query(
                state.text_for_retrieval(), res.rewritten_query
            )
            state.steps.append(
                Step("rewrite", f"第 {attempt + 1} 次重查，检索词：{state.search_query}", 0)
            )
            self._node_retrieve(state)

        # 都没判 OK → 用分数最高的那一轮，保证结果不差于第一轮
        best = max(rounds, key=lambda x: x[0]) if rounds else (0.0, [])
        if best[1]:
            state.chunks = best[1]
            state.steps.append(
                Step("judge", f"各轮均未判定充足，采用得分最高的一轮（{best[0]:.5f}）", 0)
            )
        return len(rounds) > 1

    @staticmethod
    def _top_score(chunks: list[Hit]) -> float:
        return chunks[0].score if chunks else 0.0

    def _extract_store(self, q: str) -> str | None:
        """用门店白名单匹配，不用正则猜。

        第一版用 r'[\\u4e00-\\u9fff]{2,6}店' 匹配，
        「上周广州天河店卖了多少」整段被吃成门店名「上周广州天河店」，查库必然为空。
        白名单匹配顺带解决了简称（天河店 → 广州天河店）。
        """
        try:
            res = self.tools.list_stores()
        except Exception:
            return None
        if not res.ok or not res.data:
            return None
        names = [row["name"] for row in res.data]  # type: ignore[union-attr]
        for n in sorted(names, key=len, reverse=True):
            if n in q:
                return n
            short = n[2:] if len(n) > 3 else n  # 去掉城市前缀
            if short and len(short) >= 3 and short in q:
                return n
        return None

    def _node_query_data(self, state: AgentState) -> None:
        t0 = time.perf_counter()
        q = state.query
        store_name = self._extract_store(q)
        period = _extract_period(q)
        if not store_name:
            state.steps.append(Step("query_data", "未识别到门店名，跳过结构化查询", 0))
            return
        try:
            if any(w in q for w in ("库存", "缺货", "存货", "告急")):
                res = self.tools.inventory_alert(store_name)
            elif any(w in q for w in ("核销",)):
                res = self.tools.redemption_stats(store_name, period)  # type: ignore[arg-type]
            elif any(w in q.lower() for w in ("排行", "排名", "热销", "卖得", "top")):
                res = self.tools.top_items(store_name, period)  # type: ignore[arg-type]
            else:
                res = self.tools.store_sales(store_name, period)  # type: ignore[arg-type]
            state.tool_results.append(res)
            state.steps.append(
                Step("query_data", f"工具={res.tool} store={store_name} period={period} ok={res.ok}",
                     int((time.perf_counter() - t0) * 1000))
            )
        except FileNotFoundError as e:
            state.steps.append(Step("query_data", f"跳过：{e}", 0))

    # 工具返回的字段是英文，模型得自己猜 orders 是什么意思。
    # 猜错的代价很直接：漏掉一个字段，店长就少知道一个数字。
    # 这里把已知字段翻成中文，未知字段原样保留。
    _FIELD_CN = {
        "store": "门店", "orders": "订单数", "order_count": "订单数",
        "revenue": "营业额", "amount": "金额", "avg_ticket": "客单价",
        "qty": "数量", "unit": "单位", "item_name": "品项",
        "category": "品类", "count": "数量", "total": "合计",
        "date": "日期", "day": "日期", "name": "名称", "value": "数值",
    }

    @classmethod
    def _fmt_tool_data(cls, tr: ToolResult, note: str = "") -> str:
        """把工具返回的原始结构转成人能读的文本。

        之前直接 f-string 插值 dict，模型看到的是
        `{'store': '广州天河店', 'orders': 241}` 这种 Python 字面量，
        加上工具名一起出现，模型会把「store_sales」当内容复述出来。

        每一行独立编号 [D1] [D2]：之前整个工具结果只挂一个 [D1]，
        多行数据在模型眼里是一个块，实测「库存告急」返回 2 个品项，
        答案只说了 1 个。编号必须逐行给。

        字段名中文化：模型要自己推断 orders 是什么意思，
        推断失败就漏报一个数字，而数字是这类系统唯一有价值的东西。
        """
        def fmt_row(row: dict) -> str:
            return "，".join(
                f"{cls._FIELD_CN.get(str(k), str(k))}：{v}" for k, v in row.items()
            )

        d = tr.data
        if isinstance(d, dict):
            rows = [fmt_row(d)]
        elif isinstance(d, list):
            rows = [fmt_row(row) if isinstance(row, dict) else str(row) for row in d[:8]]
        else:
            rows = [str(d)]
        tail = f"（{note}）" if note else ""
        return "\n".join(f"[D{i}] {r}{tail}" for i, r in enumerate(rows, start=1))

    @staticmethod
    def _ensure_all_rows_mentioned(answer: str, tr: ToolResult) -> str:
        """保证多行工具结果里的每一条都在答案里出现。

        实测：工具返回 2 个告急品项，模型只答了 1 个。
        改 prompt（编号、加「不要只挑一条」规则）都不管用——
        7B 模型在长列表上就是会漏。

        可靠做法是代码兜底：检查每个条目的关键标识符
        是否出现在答案里，漏了就补一句。
        宁可答案啰嗦一点，也不能少报一条——
        门店店长按这份提示去补货，漏一个品项就是真实的业务事故。
        """
        d = tr.data
        if not isinstance(d, list) or len(d) < 2:
            return answer

        missing: list[str] = []
        for row in d:
            if not isinstance(row, dict):
                continue
            name = str(row.get("item_name") or row.get("name") or "").strip()
            if not name:
                continue
            if name not in answer:
                missing.append(name)

        if not missing:
            return answer

        qty_bits = []
        for row in d:
            if isinstance(row, dict):
                name = str(row.get("item_name") or row.get("name") or "").strip()
                if name in missing:
                    qty = row.get("qty", "")
                    unit = row.get("unit", "")
                    qty_bits.append(f"{name}（{qty}{unit}）")
        extra = "、".join(qty_bits) if qty_bits else "、".join(missing)
        suffix = f"另外还有：{extra}。"
        return f"{answer.rstrip()}{suffix}"

    # 条件性强制条款：这些内容一旦命中场景就必须出现在答案里，
    # 不管模型自己有没有想起来。
    #
    # 为什么需要这个：SOP 里大量条款是条件句——「如果客人提出身体不适，
    # 立即建议就医」。问题是客人问「吃到坏了怎么办」时并没有说
    # 「身体不适」，模型读到这条会当成「当前场景不满足条件」而跳过。
    # 实测「客人吃到了坏了怎么办」连跑 5 次，「建议就医」出现 0/5。
    #
    # 这在食品安全场景是真实事故风险：漏掉就医建议，
    # 后面所有流程（退单、赔付）都建立在客人身体无恙的假设上。
    # 所以这里不靠模型自觉，按场景关键词强制补上。
    _MANDATORY_RULES: list[tuple[tuple[str, ...], str, str]] = [
        # (触发关键词, 必须出现的关键内容, 补充语句)
        (
            ("吃坏", "吃到了坏", "食物问题", "食品问题", "坏了怎么办", "吃到不对",
             "喝到坏", "变质", "异物", "酸败", "异味", "品质问题"),
            "就医",
            "如果客人出现身体不适，立即建议就医并保留就医凭证，当天上报区域运营，不得私下协商赔偿。",
        ),
    ]

    @classmethod
    def _ensure_mandatory_rules(cls, answer: str, query: str) -> str:
        out = answer
        for triggers, keyword, clause in cls._MANDATORY_RULES:
            if keyword in out:
                continue
            if not any(t in query for t in triggers):
                continue
            # 答案末尾通常带 [n] 引用标记，直接接句子会粘连成「[2]如果客人…」
            sep = "" if not out else (" " if out[-1] in "。！？.!?]" else "。")
            out = f"{out.rstrip()}{sep}{clause}"
        return out

    # 或「根据参考资料…」。这些是推理痕迹的开场白，不是答案。
    #
    # 注意：不能简单地把「首先」二字全剥掉——
    # 「首先确认是原料问题还是出品问题」是正常的中文答案开头，
    # 剥掉会破坏句子。判据是「首先」后面跟的是逗号还是正文。
    _LEAD_NOISE = (
        "首先，用户问的是", "首先,用户问的是", "首先我要回答", "首先我需要",
        "其次，", "其次,", "最后，", "最后,",
        "根据参考资料，", "根据参考资料,", "根据提供的参考资料，",
        "根据提供的参考资料,", "根据上述资料，", "根据上述资料,",
        "根据问题，", "根据问题,", "好的，", "好的,",
        "我需要根据提供的参考资料来回答。", "我需要根据参考资料来回答。",
    )

    @classmethod
    def _strip_lead_noise(cls, answer: str) -> str:
        out = answer.strip()
        changed = True
        while changed:
            changed = False
            for noise in cls._LEAD_NOISE:
                if out.startswith(noise):
                    out = out[len(noise):].strip()
                    changed = True
            # 「用户问的是：XXX。真正的答案是...」——复述问题部分也一并去掉
            m = re.match(r"^用户问的是[：:]\s*[^。！？\n]{1,60}[。！？]\s*", out)
            if m:
                out = out[m.end():].strip()
                changed = True
        return out

    # 拒答时清掉引用标记。
    # 模型有时会说「现有资料里没有相关规定」但同时留着 [1]——
    # 这自相矛盾：你不可能引用一份「没有规定」的资料。
    # 企业知识库场景下，用户点开那个引用会看到确实有相关规定，
    # 于是对整个系统的信任度直接崩掉。
    _REFUSAL_MARKERS = ("现有资料里没有相关规定", "现有资料里没有", "资料里没有相关规定")

    @classmethod
    def _strip_citations_on_refusal(cls, answer: str) -> str:
        if not any(m in answer for m in cls._REFUSAL_MARKERS):
            return answer
        return re.sub(r"\s*\[\s*(?:D?\d+)\s*\]", "", answer).strip()

    @staticmethod
    def _strip_bogus_citations(answer: str, n_chunks: int, n_data_rows: int = 0) -> str:
        """删掉指向不存在来源的引用编号。

        模型偶尔会写 [3] 而召回列表只有 2 段，或者纯数据查询
        （召回 0 段）时凭空写 [2]。这类引用编号是编的，
        点进页面会指向一个不存在的引用卡片。

        这里做硬校验：
        - [n] 只在 1..n_chunks 范围内合法
        - [Dn] 只在 1..n_data_rows 范围内合法
        宁可少一个标注，也不能给一个假的出处——
        企业知识库场景里，假引用比没引用更糟。
        """
        if n_chunks <= 0 and n_data_rows <= 0:
            # 没有任何来源，清掉所有标注
            return re.sub(r"\s*\[\s*(?:D?\d+)\s*\]", "", answer).strip()

        def keep(m: re.Match) -> str:
            raw = m.group(1)
            if raw.upper().startswith("D"):
                try:
                    n = int(raw[1:])
                except ValueError:
                    return ""
                return m.group(0) if 1 <= n <= n_data_rows else ""
            try:
                n = int(raw)
            except ValueError:
                return ""
            return m.group(0) if 1 <= n <= n_chunks else ""

        return re.sub(r"\[\s*(D?\d+)\s*\]", keep, answer).strip()

    def _node_answer(self, state: AgentState) -> None:
        t0 = time.perf_counter()
        ctx_parts: list[str] = []
        for i, h in enumerate(state.chunks, start=1):
            ctx_parts.append(f"[{i}] 《{h.doc_title} · {h.section_path}》\n{h.text}")

        # 经营数据也编号。
        # 之前经营数据是无编号的自由文本，模型看到有编号的「参考资料」、
        # 没编号的「经营数据」，就自己给数据编了个 [2]——纯引用捏造。
        # 编号之后规则统一：数据段用 [D1] 这种前缀，跟文档引用区分开，
        # 模型就不会拿文档编号去套数据。
        data_parts: list[str] = []
        for tr in state.tool_results:
            data_parts.append(self._fmt_tool_data(tr, tr.note))

        prompt = (
            f"参考资料：\n{chr(10).join(ctx_parts) or '（无）'}\n\n"
            f"经营数据：\n{chr(10).join(data_parts) or '（无）'}\n\n"
            f"问题：{state.query}"
        )
        answer = self.llm.generate(prompt, SYSTEM_PROMPT)

        if answer:
            answer = self._strip_lead_noise(answer)
            n_data_rows = sum(
                len(tr.data) if isinstance(tr.data, list) else 1
                for tr in state.tool_results
                if tr.ok and tr.data
            )
            answer = self._strip_bogus_citations(
                answer, len(state.chunks), n_data_rows
            )
            for tr in state.tool_results:
                if tr.ok:
                    answer = self._ensure_all_rows_mentioned(answer, tr)
            answer = self._ensure_mandatory_rules(answer, state.query)
            answer = self._strip_citations_on_refusal(answer)
            state.answer = answer
            state.steps.append(
                Step("generate", f"llm={self.llm.name} chars={len(answer)}",
                     int((time.perf_counter() - t0) * 1000))
            )
        else:
            state.degraded = True
            state.answer = self._extractive_answer(state)
            state.steps.append(
                Step("generate", "llm 不可用，走抽取式兜底", int((time.perf_counter() - t0) * 1000))
            )

        state.citations = [
            {
                "index": i,
                "doc_id": h.doc_id,
                "doc_title": h.doc_title,
                "section_path": h.section_path,
                "source": h.source,
                "score": round(h.score, 5),
                "text": h.text[:300],
            }
            for i, h in enumerate(state.chunks, start=1)
        ]

    def _extractive_answer(self, state: AgentState) -> str:
        ok_data = [t for t in state.tool_results if t.ok and t.data]
        failed = [t for t in state.tool_results if not t.ok]

        if not state.chunks and not ok_data:
            msg = "现有资料里没有找到相关内容。"
            if failed:
                msg += f"结构化查询也没拿到数据（{failed[0].note}）。"
            msg += " 建议确认一下门店名称或时间范围，或直接问店长。"
            return msg

        lines: list[str] = []
        if ok_data:
            tr = ok_data[0]
            lines.append(f"经营数据（{tr.note}）：")
            if isinstance(tr.data, dict):
                for k, v in tr.data.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(tr.data, list):
                for row in tr.data[:5]:
                    lines.append("  - " + "，".join(f"{k}={v}" for k, v in row.items()))
            lines.append("")
        if failed:
            lines.append(f"（{failed[0].tool} 未查到数据：{failed[0].note}）")
            lines.append("")
        if state.chunks:
            lines.append("资料依据：")
            for i, h in enumerate(state.chunks[:3], start=1):
                body = h.text.replace("\n", " ").strip()
                if len(body) > 220:
                    body = body[:220] + "…"
                lines.append(f"[{i}] {h.doc_title} · {h.section_path}")
                lines.append(f"    {body}")
                lines.append("")
            lines.append("（当前未接入生成模型，以上为原文摘录。配置 SM_LLM_BACKEND 后可直接成文。）")
        return "\n".join(lines).strip()

    # ---------- 运行 ----------

    def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        self._node_route(state)
        if state.intent in (Intent.POLICY, Intent.BOTH):
            self._node_retrieve(state)
        if state.intent in (Intent.DATA, Intent.BOTH):
            self._node_query_data(state)
        # 两路汇合：数据查询没结果时回落知识库
        self._node_merge_fallback(state)
        # Agent 层：自检 → 不够就改写重查
        self._node_judge_retry(state)
        self._node_answer(state)
        return state
