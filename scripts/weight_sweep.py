"""向量权重对照实验。

为什么需要这个脚本：
最初只有字符哈希嵌入（hashing）。实测发现向量路在混合检索里是负资产——
BM25 单独把正解排第 1，向量路却把无关片段排第 1，RRF 奖励「两路都出现」
反而把正解稀释到第 3。于是把向量权重压到 0.15。

换到 bge-m3 之后，这个前提变了：向量路现在有真实语义能力，
0.15 这个值就不再成立。但「不再是负资产」不等于「该设成 1.0」——
它可能是个中间值。唯一能回答的办法是扫一遍，用同一份金标集量出来。

跑法：
    python scripts/weight_sweep.py            # 扫权重，输出 recall@k 表
    python scripts/weight_sweep.py --backend hashing   # 换后端再扫一遍
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
from app.rag.retriever import _path_bonus, expand_query, rrf_fuse  # noqa: E402
from app.rag.store import build_store  # noqa: E402

WEIGHTS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_cases() -> list[dict]:
    data = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def run_sweep(store: Store, embedder, cases: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for w in WEIGHTS:
        hit1 = hit3 = hit5 = total = 0
        for c in cases:
            if c["type"] != "policy":
                continue
            total += 1
            q = expand_query(c["query"])
            qv = embedder.embed_one(q)
            vec = store.vector_search(qv, 20)
            bm = store.bm25_search(q, 20)
            fused = rrf_fuse([vec, bm], k=60, weights=[w, 1.0])
            # 路径加权重排是线上真实链路的一环，必须算进来，
            # 否则 sweep 的数字和 eval.py 的对不上，两份数据没法互相印证。
            for h in fused:
                h.score += _path_bonus(q, h.section_path)
            fused.sort(key=lambda x: x.score, reverse=True)
            paths = [h.section_path for h in fused]
            expect = c.get("expect_section", "")
            # 用子串包含而不是 endswith，跟 eval.py 保持一致。
            # 金标里存的是章节名（如「酒水异味」），实际路径是
            # 「客诉处理手册 > 酒水异味或品质问题」，endswith 会漏判。
            if any(expect in p for p in paths[:1]):
                hit1 += 1
            if any(expect in p for p in paths[:3]):
                hit3 += 1
            if any(expect in p for p in paths[:5]):
                hit5 += 1
        rows.append({
            "weight": w,
            "recall@1": round(hit1 / total * 100, 1) if total else 0,
            "recall@3": round(hit3 / total * 100, 1) if total else 0,
            "recall@5": round(hit5 / total * 100, 1) if total else 0,
            "n": total,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="", help="覆盖嵌入后端（hashing / ollama）")
    ap.add_argument("--out", default="", help="写入 UTF-8 文件")
    args = ap.parse_args()

    settings.ensure_dirs()
    backend = args.backend or settings.embed_backend
    store = build_store(settings)
    embedder = build_embedder(
        backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )
    cases = load_cases()

    lines: list[str] = []
    lines.append(f"嵌入后端  : {embedder.name} (dim={embedder.dim})")
    lines.append(f"规则题数量: {sum(1 for c in cases if c['type'] == 'policy')}")
    lines.append("")
    lines.append("| 向量权重 | recall@1 | recall@3 | recall@5 |")
    lines.append("|---|---|---|---|")

    rows = run_sweep(store, embedder, cases)
    for r in rows:
        lines.append(f"| {r['weight']} | {r['recall@1']}% | {r['recall@3']}% | {r['recall@5']}% |")

    best = max(rows, key=lambda r: (r["recall@1"], r["recall@3"]))
    lines.append("")
    lines.append(f"最优权重（按 recall@1，其次 recall@3）: {best['weight']}"
                 f"  recall@1={best['recall@1']}%  recall@3={best['recall@3']}%")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
