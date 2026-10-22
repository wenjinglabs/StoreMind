"""意图路由：判断该查知识库、查数据库，还是两个都查。

设计取舍：规则优先，LLM 兜底。

面试被问「为什么不用 LLM 直接分类」，答案是：门店场景的高频问题高度模式化，
「能……吗 / 怎么 / 规定」这类句式用正则判断的准确率高于小模型，而且零成本、零延迟。
规则命中不了才走 LLM——不是所有环节都值得塞一次模型调用。

第一版用关键词包含判断，踩了个坑：「会员卡能跨店用吗」不含「能不能」三个字，
所以没命中。改成句式正则后准确率明显上去。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    POLICY = "policy"   # 规则、流程、标准 → 走知识库
    DATA = "data"       # 事实、统计、数字 → 走数据库
    BOTH = "both"       # 两者都要


@dataclass
class RouteResult:
    intent: Intent
    confidence: float
    reason: str


POLICY_PATTERNS = [
    (r"能[^，。？!]{0,10}[吗嘛]", "疑问句式「能…吗」"),
    (r"可[^，。？!]{0,8}[吗嘛]", "疑问句式「可…吗」"),
    (r"[需用要][^，。？!]{0,8}[吗嘛]", "疑问句式「需要…吗」"),
    (r"怎么[办样做]?", "疑问词「怎么」"),
    (r"如何", "疑问词「如何」"),
    (r"为什么", "疑问词「为什么」"),
    (r"(?:什么|哪些)(?:原因|情况|要求|标准|规定|条件|东西|材料|流程|区别)", "疑问词+名词"),
    (r"(?:注意|需要|要)(?:什么|哪些|带|准备)", "疑问词「注意/要什么」"),
    (r"是否", "疑问词「是否」"),
    (r"(?:规定|流程|标准|要求|制度|规则|政策|办法|红线)", "制度类名词"),
    (r"多久|多长时间|期限|有效期", "时效类提问"),
]

# 强信号：出现即可判定为查询经营数据（这些词只可能来自数据类问题）
DATA_STRONG = [
    (r"(?:销量|营业额|营收|客单价|库存|订单数|流水|毛利)", "经营指标"),
]

# 弱信号：单独出现不足以判定。
# 「手续费多少」里的「多少」只是口语，如果按单个弱信号判 data，
# 会被路由到数据库、然后因为没有门店名而空跑一趟。
DATA_WEAK = [
    (r"多少(?:单|钱|元|杯|瓶|个|次|家|呢|？|\?)?", "数量提问"),
    (r"几(?:个|号|笔|单|家|次|瓶)", "计数提问"),
    (r"(?:上周|昨天|昨日|前天|今天|今日|本月|这个月|当月|近七天|最近七天)", "时间范围"),
    (r"(?:核销|排行|排名|热销|卖得|top\s*\d*|最高|最低|前\s*[0-9一二三四五六七八九十]+)", "指标词"),
    (r"(?:对比|比较|趋势|占比|同比|环比)", "对比分析"),
]

COMPARE_RE = re.compile(r"(?:对比|比较|趋势|哪个更|排名|排行|占比)")
STORE_HINT_RE = re.compile(r"[\u4e00-\u9fff]{2,6}店")


def _match(patterns: list[tuple[str, str]], q: str) -> list[str]:
    """按实际命中次数计，不按 pattern 计。

    第一版每个 pattern 最多记一次，结果「深圳福田店热销排行前五」里
    热销、排行、前五三个词都落在同一个 pattern 上，只算 1 个弱信号，
    不足 2 个的门槛，被误判成规则意图、跳过了数据查询。
    """
    hits: list[str] = []
    for pat, label in patterns:
        found = re.findall(pat, q, flags=re.IGNORECASE)
        if found:
            hits.extend([label] * len(found))
    return hits


def classify(query: str) -> RouteResult:
    q = (query or "").strip()
    if not q:
        return RouteResult(Intent.POLICY, 0.0, "空查询，默认走知识库")

    p_hits = _match(POLICY_PATTERNS, q)
    strong = _match(DATA_STRONG, q)
    weak = _match(DATA_WEAK, q)
    has_compare = bool(COMPARE_RE.search(q))
    has_store = bool(STORE_HINT_RE.search(q))

    is_data = bool(strong) or len(weak) >= 2

    if p_hits and is_data:
        if has_compare or has_store:
            return RouteResult(
                Intent.BOTH, 0.8,
                f"规则信号({p_hits[0]}) + 数据信号({(strong + weak)[0]}) + 门店/对比 → 两路都要",
            )
        return RouteResult(
            Intent.POLICY, 0.65, f"规则信号({p_hits[0]}) 优先于数据信号({(strong + weak)[0]})"
        )

    if is_data:
        conf = 0.85 if strong else min(0.6 + 0.1 * len(weak), 0.8)
        return RouteResult(Intent.DATA, conf, "数据信号: " + "、".join((strong + weak)[:3]))

    if p_hits:
        return RouteResult(Intent.POLICY, min(0.6 + 0.1 * len(p_hits), 0.95),
                           "规则信号: " + "、".join(p_hits[:3]))

    return RouteResult(Intent.POLICY, 0.35, "无明确信号，默认知识库")
