"""dump judge 模型的原始输出，看解析为什么失败。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.judge import EvidenceJudge
from app.config import settings
from app.rag.embedder import build_embedder
from app.rag.retriever import expand_query, hybrid_search
from app.rag.store import build_store
from app.agent.llm import build_llm
from app.agent.judge import JUDGE_SYSTEM

settings.ensure_dirs()
store = build_store(settings)
emb = build_embedder(settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model)
llm = build_llm(settings.llm_backend, ollama_base=settings.ollama_base,
                model=settings.llm_model, base=settings.openai_base,
                key=settings.openai_key, timeout=settings.llm_timeout)

QS = ["酒里有絮状物", "经理说要给我打折", "客人拍了视频发到网上", "不想用了能退钱吗"]

out = []
for q in QS:
    chunks = hybrid_search(store, emb, expand_query(q), top_k=5, vector_k=20, bm25_k=20, rrf_k=60)
    ctx = "\n".join(
        f"[{i}] 《{h.section_path}》{h.text[:220]}"
        for i, h in enumerate(chunks[:4], start=1)
    )
    prompt = f"资料片段：\n{ctx}\n\n问题：{q}"
    raw = llm.generate(prompt, JUDGE_SYSTEM, temperature=0.0)
    out.append("=" * 72)
    out.append(f"Q: {q}")
    out.append("--- 原始输出 ---")
    out.append(repr(raw))
    out.append("--- 解析后 ---")
    lines = [ln.strip() for ln in (raw or "").strip().splitlines() if ln.strip()]
    out.append(f"  行数={len(lines)}")
    if lines:
        out.append(f"  head={lines[0]!r}  是否OK={lines[0].upper().startswith('OK')}  是否MISS={lines[0].upper().startswith('MISS')}")
    if len(lines) > 1:
        out.append(f"  second={lines[1]!r}")
    else:
        out.append("  second=<缺失>")

text = "\n".join(out)
Path("data/judge_raw.txt").write_text(text, encoding="utf-8")
print("written data/judge_raw.txt")
