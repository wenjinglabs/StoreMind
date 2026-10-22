"""标定拒答阈值：比较命中组与无答案组的 BM25 原始分。

RRF 分数不能用来拒答——它只反映排名，榜单第 1 名永远约 1/61，
跟问题相不相关无关。实测命中组最低 0.02013、无答案组最高 0.02007，完全重叠。
BM25 原始分带 idf 加权，关键词没命中时分数会真的掉下去，才是可用的判据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.retriever import expand_query  # noqa: E402
from app.rag.store import build_store  # noqa: E402

cases = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))

settings.ensure_dirs()
store = build_store(settings)
embedder = build_embedder("hashing", ollama_base="", model="")

rows = []
for c in cases:
    q = expand_query(c["query"])
    hits = store.bm25_search(q, 1)
    score = round(hits[0].score, 2) if hits else 0.0
    rows.append((c["id"], c["type"], score, hits[0].section_path if hits else "-"))

hit = [r for r in rows if r[1] == "policy"]
noa = [r for r in rows if r[1] == "noanswer"]

lines = ["== policy（应命中）=="]
for r in hit:
    lines.append(f"  {r[0]}  bm25_top1={r[2]:>8.2f}  {r[3]}")
lines.append("")
lines.append("== noanswer（应拒答）==")
for r in noa:
    lines.append(f"  {r[0]}  bm25_top1={r[2]:>8.2f}  {r[3]}")
lines.append("")
lines.append(f"policy 最低分 : {min(r[2] for r in hit):.2f}")
lines.append(f"noans  最高分 : {max(r[2] for r in noa):.2f}")
lines.append(f"间隔          : {min(r[2] for r in hit) - max(r[2] for r in noa):.2f}")

text = "\n".join(lines)
(ROOT / "data" / "_threshold.txt").write_text(text, encoding="utf-8")
print(text)
