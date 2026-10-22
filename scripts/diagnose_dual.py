"""双路召回诊断：看向量路和 BM25 路各自召回了什么。

用途：验证「混合检索里两路到底各贡献了什么」。
weight_sweep 显示把向量权重压到 0 指标不变，这不常见——
正常情况下两路应该有互补性，否则就不叫混合检索了。
这个脚本用来定位是「向量路真的没贡献」还是「指标不够敏感」。
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
from app.rag.retriever import expand_query, rrf_fuse, _path_bonus  # noqa: E402
from app.rag.store import build_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    settings.ensure_dirs()
    store = build_store(settings)
    embedder = build_embedder(
        settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )
    cases = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    pol = [c for c in cases if c["type"] == "policy"]

    lines: list[str] = []
    lines.append(f"embedder = {embedder.name}  dim={embedder.dim}")
    lines.append("")

    stats = {"vec_hit1": 0, "bm_hit1": 0, "both_hit1": 0,
             "only_vec": 0, "only_bm": 0, "neither": 0}

    for c in pol:
        q = expand_query(c["query"])
        expect = c["expect_section"]
        qv = embedder.embed_one(q)
        vec = store.vector_search(qv, 20)
        bm = store.bm25_search(q, 20)

        v_top = [h.section_path for h in vec[:args.top]]
        b_top = [h.section_path for h in bm[:args.top]]
        v_hit = any(expect in p for p in v_top)
        b_hit = any(expect in p for p in b_top)

        if v_hit:
            stats["vec_hit1"] += 1
        if b_hit:
            stats["bm_hit1"] += 1
        if v_hit and b_hit:
            stats["both_hit1"] += 1
        elif v_hit:
            stats["only_vec"] += 1
        elif b_hit:
            stats["only_bm"] += 1
        else:
            stats["neither"] += 1

        lines.append(f"[{c['id']}] {c['query']}   期望章节含「{expect}」")
        lines.append(f"   向量路 top1: {v_top[0] if v_top else '-'}   {'✔命中' if v_hit else '✘未中'}")
        lines.append(f"   BM25  top1: {b_top[0] if b_top else '-'}   {'✔命中' if b_hit else '✘未中'}")
        # 向量路 top3 全列出来，看正解排在第几
        vr = next((i + 1 for i, p in enumerate(v_top) if expect in p), None)
        br = next((i + 1 for i, p in enumerate(b_top) if expect in p), None)
        lines.append(f"   正解排名: 向量路={vr if vr else '>5'}  BM25={br if br else '>5'}")
        lines.append("")

    n = len(pol)
    lines.append("=" * 60)
    lines.append(f"规则题总数: {n}")
    lines.append(f"向量路 top{args.top} 命中: {stats['vec_hit1']} ({stats['vec_hit1']/n*100:.1f}%)")
    lines.append(f"BM25  top{args.top} 命中: {stats['bm_hit1']} ({stats['bm_hit1']/n*100:.1f}%)")
    lines.append(f"两路都命中: {stats['both_hit1']}")
    lines.append(f"仅向量命中: {stats['only_vec']}")
    lines.append(f"仅BM25命中: {stats['only_bm']}")
    lines.append(f"两路都没中: {stats['neither']}")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
