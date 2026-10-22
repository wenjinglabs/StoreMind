"""混合检索：向量召回 + 关键词召回，用 RRF 融合。

为什么不能只用向量：
门店语料里有大量型号、编号、专有名词（「黑牌」是威士忌不是颜色、「连娱宝」是产品名）。
这类词在嵌入空间里和字面完全不相关，向量召回经常漏掉。
反过来，关键词召回对「那如果客人喝了半瓶怎么办」这种口语化长句几乎无效。

两个都召，用 RRF 融合——RRF 只看排名不看分数，天然免疫两种分数量纲不同的问题。
"""
from __future__ import annotations

from .store import Hit, tokenize
from .pg_store import PgStore

# 口语 → 术语映射。这是「词汇鸿沟」的最小成本解。
# 实测案例：「满减和折扣券能一起用吗」在原始 query 下，正解「叠加规则」
# 只排第 3 位——因为文档说「不能叠加」，客人说「能一起用」，字面零重叠。
# 正规解法是换语义更强的嵌入模型；在字符级检索下，同义词扩展是性价比最高的补丁。
EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (("一起用", "同时用", "都用", "叠加用", "共用"), "叠加"),
    (("别的店", "其他店", "另一家店", "别家", "外店", "跨门店"), "跨店"),
    (("送的", "赠送的", "赠的", "送的钱"), "赠送"),
    (("过期", "失效", "超期", "用不了"), "有效期"),
    (("带什么", "要带", "需要带", "准备什么"), "携带"),
    (("卖了多少", "卖了多少钱", "做了多少", "营收"), "营业额"),
    (("能退", "怎么退", "退掉", "不想要了"), "退卡"),
    (("没货", "缺货", "不够卖", "告急"), "库存"),
    (("投诉", "不高兴", "生气", "不满意"), "客诉"),
]


def expand_query(query: str) -> str:
    """把口语词展开成术语，拼在原 query 后面。

    只追加不替换：原 query 的字面信息对 BM25 有用，不能丢。
    """
    extra: list[str] = []
    for words, term in EXPANSIONS:
        if any(w in query for w in words) and term not in query:
            extra.append(term)
    return f"{query} {' '.join(extra)}" if extra else query


# 这些词出现在 query 里说明用户在问规则，不是问数据
RULE_HINTS = ("怎么", "如何", "能不能", "可以吗", "规定", "流程", "标准", "要求", "为什么", "是否")


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """加权 RRF。weights 与 rankings 一一对应。

    为什么需要权重：hashing 嵌入只是字符 n-gram 哈希，语义能力有限；
    在中文短文档场景，BM25 的字面匹配通常比它更靠谱。
    所以 hashing 后端下压低向量权重，切到 bge-m3 后再提回 1.0。
    这是可解释的取舍，不是玄学调参。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    agg: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for hits, w in zip(rankings, weights):
        for rank, h in enumerate(hits, start=1):
            agg[h.chunk_id] = agg.get(h.chunk_id, 0.0) + w / (k + rank)
            if h.chunk_id not in best:
                best[h.chunk_id] = h
    merged = []
    for cid, score in agg.items():
        h = best[cid]
        merged.append(
            Hit(h.chunk_id, h.doc_id, h.doc_title, h.section_path, h.text, score, "hybrid")
        )
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def _path_bonus(query: str, section_path: str) -> float:
    """轻量重排：query 词命中章节标题的加一点分。

    没有上 rerank 模型（bge-reranker 要额外一次推理），
    但这个信号在 SOP 场景特别有效——章节标题就是人写的标签。
    """
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    p_terms = set(tokenize(section_path))
    overlap = len(q_terms & p_terms)
    return min(overlap, 6) * 0.0015


def hybrid_search(store: PgStore, embedder, query: str, *, top_k: int = 5,
                  vector_k: int = 20, bm25_k: int = 20, rrf_k: int = 60,
                  vector_weight: float | None = None) -> list[Hit]:
    q_expanded = expand_query(query)
    if vector_weight is None:
        # 实测结论：hashing 后端下向量路是负资产。
        # 「满减和折扣券能一起用吗」这一条，BM25 把正解「叠加规则」排第 1（40.9 分，
        # 领先第二名 60%），向量路却把「抖音素材留存」排第 1，融合后正解掉到第 3。
        # RRF 会奖励「两路都出现」的片段，所以弱向量路的副作用不是加噪声，是稀释。
        # 降到 0.15 保留一点同义改写的语义信号，但让字面匹配主导。
        # 换 bge-m3 之后这个值要回到 1.0 并重新跑评估集——这是待验事项，不是定论。
        vector_weight = 0.15 if getattr(embedder, "name", "") == "hashing" else 1.0
    qv = None
    try:
        qv = embedder.embed_one(q_expanded)
    except Exception:
        # 嵌入服务不可用时不能让整条问答 500：BM25 仍然能工作。
        # 这是有意的降级，不是静默失败——trace 里会看到 vector 路为空。
        qv = None
    vec_hits = store.vector_search(qv, vector_k) if qv else []
    bm_hits = store.bm25_search(q_expanded, bm25_k)
    fused = rrf_fuse([vec_hits, bm_hits], k=rrf_k, weights=[vector_weight, 1.0])
    for h in fused:
        h.score += _path_bonus(q_expanded, h.section_path)
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]


def debug_search(store: PgStore, embedder, query: str, k: int = 5) -> dict:
    """诊断用：分别看两路召回，定位是哪一路把正确结果丢了。"""
    q_expanded = expand_query(query)
    try:
        qv = embedder.embed_one(q_expanded)
        vector = [
            {"section": h.section_path, "score": round(h.score, 5)}
            for h in store.vector_search(qv, k)
        ]
    except Exception:
        vector = []
    return {
        "query": query,
        "expanded": q_expanded,
        "vector": vector,
        "bm25": [
            {"section": h.section_path, "score": round(h.score, 5)}
            for h in store.bm25_search(q_expanded, k)
        ],
        "hybrid": [
            {"section": h.section_path, "score": round(h.score, 5)}
            for h in hybrid_search(store, embedder, query, top_k=k)
        ],
    }
