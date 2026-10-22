"""证据充分性判定 + 查询改写。

这是「Agent」和「流水线」的分界线。

流水线：检索到什么都直接拿去生成，生成什么算什么。
Agent：先自检——手上这些证据够不够回答这个问题。
        不够 → 改写检索词 → 重查 → 再自检，直到够或者次数用完。

为什么必须要有这一层（三个实测理由）：

1. **检索永远返回 top5，哪怕全是不相关内容。**
   没有自检，模型就会拿着 5 段无关资料硬答，看起来很自信，实际是编的。

2. **用户提问和文档用词不一致时，第一次检索会漏。**
   客人说「吃到变质的酒」，文档写「酒水异味或品质问题」。
   第一次召回的 top1 未必是正确段落，但只要自检发现「这些资料答不了」
   并把检索词改写成语料里的说法，第二次就能召回到。

3. **README 里那个拒答率 0% → 100% 的结论，本质就是自检。**
   检索分判断不了「资料够不够回答」，那是语义判断，必须由模型做。

设计上刻意分成两层：
- **规则层**（快、零成本）：工具拿到数据 → 直接算够。词面重合度够高 → 算够。
- **模型层**（准、要花钱）：规则层说不清楚时，才叫 LLM 判断。

这样常见问题不付 LLM 的钱，只有疑难问题才走模型。这跟真实生产系统的做法一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..rag.store import Hit, tokenize


def merge_hits(old: list[Hit], new: list[Hit], limit: int = 5) -> list[Hit]:
    """两轮召回取并集，同一 chunk 保留分数更高的版本。

    为什么合并而不是覆盖：改写后的检索词可能召回了更好的片段，也可能反而更差。
    取并集保证「重查永远不会让结果变差」——最坏情况是多几个候选，
    不会把第一轮已经找到的正确答案弄丢。
    """
    by_id: dict[int, Hit] = {}
    for h in old:
        by_id[h.chunk_id] = h
    for h in new:
        prev = by_id.get(h.chunk_id)
        if prev is None or h.score > prev.score:
            by_id[h.chunk_id] = h
    out = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
    return out[:limit]


# 模型层判定用的提示词。
# 要求严格两行输出，便于解析——解析失败就退回规则层，不阻塞主链路。
#
# 这里踩过一个坑：第一版提示词里写了「OK 时第二行写 -」，
# 结果模型在 MISS 时也照抄「MISS\n-」，改写词永远是空的，
# 整个重查机制等于空转（实测 10 条用例，重查修正 top1 的数量是 0）。
# 小模型会照抄格式示例，所以示例不能只给一种情况。
JUDGE_SYSTEM = """你是检索质量评估员。判断给定的资料片段能否回答用户的问题。

输出格式（必须严格遵守，只输出两行）：
第一行：OK 或 MISS
第二行：改写后的检索词（20 字以内，用更贴近资料原文的正式说法）

第二行的填法：
- 如果第一行是 OK，第二行写「无」
- 如果第一行是 MISS，第二行必须写一个真正不同的检索词。
  例如问题说「酒里有絮状物」，资料里写「酒水异味或品质问题」，
  第二行就写「酒水异味 品质问题」。

判断标准：
- 资料里有明确的规定、流程、数据，就是 OK
- 资料只是沾边但不包含答案，或者完全无关，就是 MISS
- 宁可判 MISS，也不要因为资料看起来相关就判 OK
- 第二行绝对不能写「-」或「无」，除非第一行是 OK"""


@dataclass
class JudgeResult:
    sufficient: bool
    reason: str
    rewritten_query: str = ""
    used_model: bool = False
    # 自检轨迹，进 trace 让「为什么重试」可见
    details: dict = field(default_factory=dict)


def _term_overlap(query: str, chunks: list[Hit]) -> float:
    """query 的实词有多少出现在 top1 片段里。

    用字符 bigram + 单字（跟 store.tokenize 一致），不做分词器依赖。
    0 表示完全没共同词，1 表示 query 的词全在片段里。
    """
    if not query or not chunks:
        return 0.0
    q = set(tokenize(query))
    if not q:
        return 0.0
    top = set(tokenize(chunks[0].text))
    # 章节标题也算，标题是强标签
    top |= set(tokenize(chunks[0].section_path))
    return len(q & top) / len(q)


class EvidenceJudge:
    """证据充分性判定器。

    规则层先行，模型层兜底。模型层只在规则层说不清楚时才调用。
    """

    # 词面重合度高于此值，直接判够，不问模型
    STRONG_OVERLAP = 0.55
    # 低于此值，直接判不够（几乎没共同词，重查有意义）
    WEAK_OVERLAP = 0.15

    def __init__(self, llm=None, max_rewrite_len: int = 40) -> None:
        self.llm = llm
        self.max_rewrite_len = max_rewrite_len
        # 规则层用的章节标题表：改写词的兜底来源。
        # 模型给不出改写词时，从这里挑语义最接近的章节标题当检索词。
        self._section_titles: list[str] = []
        self._title_tokens: list[set[str]] = []

    def load_sections(self, store) -> None:
        """把语料的文档名和章节标题都读进来，作为改写词的候选池。

        为什么要兜底：小模型经常只输出「MISS」不给改写词
        （第一版提示词里示例写「OK 时输出 -」，模型在 MISS 时也照抄）。
        没有兜底的话整个重查机制就是空转。

        两个层级都要收：文档名（「会员卡与储值规则」）和
        章节名（「退卡与转让」）。只收文档名的话，
        「不想用了能退钱吗」这种问法匹配不到任何文档名，兜底会退化成原句。
        """
        titles: list[str] = []
        try:
            for row in store.list_docs():
                title = row.get("title") if isinstance(row, dict) else None
                if title:
                    titles.append(str(title))
                doc_id = row.get("doc_id") if isinstance(row, dict) else None
                if doc_id:
                    for ch in store.get_doc_chunks(doc_id):
                        sp = str(ch.get("section_path") or "")
                        if " > " in sp:
                            # 章节全路径和叶子名都进池
                            titles.append(sp)
                            titles.append(sp.split(" > ")[-1])
                        elif sp:
                            titles.append(sp)
        except Exception:
            pass

        seen: set[str] = set()
        clean: list[str] = []
        for t in titles:
            t = t.strip()
            if len(t) >= 2 and t not in seen:
                seen.add(t)
                clean.append(t)
        self._section_titles = clean
        self._title_tokens = [set(tokenize(t)) for t in clean]

    @staticmethod
    def compose_query(original: str, rewritten: str) -> str:
        """把改写词和原句拼在一起，而不是替换。

        替换会丢掉用户原话里的信号：「客人吃到变质的酒怎么办」改写成
        「酒水异味 品质问题」之后，「变了」「客人」这些词没了，
        而这些词可能正好在另一个相关章节里。
        拼接让 BM25 同时吃到两边的词，语义路也不受影响。
        """
        if not rewritten or rewritten == original:
            return original
        if rewritten in original:
            return original
        return f"{rewritten} {original}"

    def _fallback_rewrite(self, query: str, chunks: list[Hit]) -> str:
        """规则兜底：找语义最接近的章节标题当检索词。

        做法：对每个候选标题，算它跟原 query 的 token 重合度，
        取重合度最高且大于 0 的那个。重合度为 0 就没得挑，原样返回。
        """
        if not self._section_titles:
            return query
        q = set(tokenize(query))
        if not q:
            return query
        best = ""
        best_score = 0.0
        for title, toks in zip(self._section_titles, self._title_tokens):
            if not toks:
                continue
            s = len(q & toks) / len(q | toks)
            if s > best_score:
                best_score = s
                best = title
        return best or query

    # ---------- 规则层 ----------

    def _rule_judge(self, query: str, chunks: list[Hit], has_tool_data: bool) -> JudgeResult | None:
        """返回 None 表示规则层说不清楚，交给模型层。"""
        if has_tool_data:
            return JudgeResult(True, "工具已返回数据，无需额外检索", details={"overlap": None})
        if not chunks:
            return JudgeResult(False, "没有召回任何片段", details={"overlap": 0.0})
        ov = _term_overlap(query, chunks)
        if ov >= self.STRONG_OVERLAP:
            return JudgeResult(True, f"top1 词面重合度 {ov:.2f}，直接作答",
                               details={"overlap": round(ov, 3)})
        if ov <= self.WEAK_OVERLAP:
            return JudgeResult(False, f"top1 词面重合度 {ov:.2f}，几乎无关，建议改写重查",
                               details={"overlap": round(ov, 3)})
        return None  # 灰色地带，交给模型判断

    # ---------- 模型层 ----------

    def _model_judge(self, query: str, chunks: list[Hit]) -> JudgeResult:
        if self.llm is None or getattr(self.llm, "name", "none") == "none":
            return JudgeResult(True, "无可用模型，保守作答", details={"model": "unavailable"})

        ctx = "\n".join(
            f"[{i}] 《{h.section_path}》{h.text[:220]}"
            for i, h in enumerate(chunks[:4], start=1)
        )
        prompt = f"资料片段：\n{ctx}\n\n问题：{query}"

        try:
            raw = self.llm.generate(prompt, JUDGE_SYSTEM, temperature=0.0)
        except Exception:
            return JudgeResult(True, "判定异常，保守作答", details={"model": "error"})
        if not raw:
            return JudgeResult(True, "判定无输出，保守作答", details={"model": "empty"})

        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            return JudgeResult(True, "判定无输出，保守作答", details={"model": "empty"})

        head = lines[0].upper()
        rewritten = ""
        if len(lines) > 1:
            cand = lines[1].strip()
            # 「-」「无」「没有」都表示没给改写词
            if cand and cand not in ("-", "—", "无", "没有", "N/A", "na"):
                rewritten = cand[: self.max_rewrite_len].strip()

        if head.startswith("MISS"):
            # 模型没给改写词 → 规则兜底，否则重查会用同一个词，等于空转
            if not rewritten:
                rewritten = self._fallback_rewrite(query, chunks)
            return JudgeResult(
                False,
                "模型判定资料不足以回答",
                rewritten_query=rewritten,
                used_model=True,
                details={"model_raw": raw[:120]},
            )
        return JudgeResult(True, "模型判定资料充足",
                           used_model=True, details={"model_raw": raw[:120]})

    # ---------- 对外 ----------

    def judge(self, query: str, chunks: list[Hit], has_tool_data: bool = False) -> JudgeResult:
        r = self._rule_judge(query, chunks, has_tool_data)
        if r is not None:
            return r
        return self._model_judge(query, chunks)
