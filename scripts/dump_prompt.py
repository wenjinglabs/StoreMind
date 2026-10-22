"""看真实送给模型的 prompt，定位「工具返回了数据但回答说没有」。

上一个问题：问「库存告急」，工具返回了 2 条数据，模型却答
「现有资料里没有相关规定」。要么数据没进 prompt，要么进了但模型看不懂。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.pipeline import Pipeline, SYSTEM_PROMPT
from app.agent.llm import build_llm
from app.config import settings
from app.rag.embedder import build_embedder
from app.rag.store import build_store
from app.tools.ops_tools import build_ops_tools

settings.ensure_dirs()
store = build_store(settings)
emb = build_embedder(settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model)
llm = build_llm(settings.llm_backend, ollama_base=settings.ollama_base,
                model=settings.llm_model, base=settings.openai_base,
                key=settings.openai_key, timeout=settings.llm_timeout)
tools = build_ops_tools(settings)
pipe = Pipeline(store, emb, llm, tools, settings)

out: list[str] = []

for q in ["广州天河店库存告急的有哪些", "储值2000送多少钱", "上周广州天河店卖了多少"]:
    st = pipe.run(q)
    out.append("=" * 72)
    out.append(f"Q: {q}   intent={st.intent.value}  chunks={len(st.chunks)}  tools={len(st.tool_results)}")
    out.append("--- 格式化后的经营数据（就是进 prompt 的那段）---")
    for tr in st.tool_results:
        out.append(f"  raw={tr.data}")
        out.append(f"  fmt={pipe._fmt_tool_data(tr)}")
    out.append(f"--- 答案 ---\n{st.answer}")
    out.append("")

# 顺便看完整 prompt 长什么样
st = pipe.run("广州天河店库存告急的有哪些")
ctx_parts = []
for i, h in enumerate(st.chunks, start=1):
    ctx_parts.append(f"[{i}] 《{h.doc_title} · {h.section_path}》\n{h.text}")
data_parts = []
for j, tr in enumerate(st.tool_results, start=1):
    data_parts.append(f"[D{j}] {pipe._fmt_tool_data(tr)}\n（{tr.note}）")
prompt = (f"参考资料：\n{chr(10).join(ctx_parts) or '（无）'}\n\n"
          f"经营数据：\n{chr(10).join(data_parts) or '（无）'}\n\n"
          f"问题：{st.query}")
out.append("=" * 72)
out.append("=== 完整 prompt（库存题）===")
out.append(prompt)
out.append("")
out.append("=== 系统提示词 ===")
out.append(SYSTEM_PROMPT)

Path("data/prompt_dump.txt").write_text("\n".join(out), encoding="utf-8")
print("ok")
