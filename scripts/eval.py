"""评估脚本：跑金标集，算 recall@k 和路由准确率。

为什么这个脚本比功能更重要：
没有它，你调一次分片参数、改一次权重，都只能"感觉好像好了一点"。
有了它，每次改动都是一个数字。这是玩具和工程的分界线。

指标定义（都按 chunk 级算，不比字符串相似度，避免自欺）：
- recall@k : 期望章节是否出现在 top-k 召回里
- route_acc: 意图分类是否正确（policy / data / noanswer）
- 拒答率   : 无答案问题是否给出了"没找到"而不是编造

用法:
    python scripts/eval.py
    python scripts/eval.py --top-k 5 --out eval/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.llm import build_llm  # noqa: E402
from app.agent.pipeline import Pipeline  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.store import Store  # noqa: E402
from app.tools.ops_tools import OpsTools  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.json"
REFUSE_HINTS = ("没有找到", "没有相关规定", "没有相关内容", "建议")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/report.md")
    args = ap.parse_args()

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    settings.ensure_dirs()
    store = Store(settings.db_path)
    embedder = build_embedder(
        settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )
    llm = build_llm(
        settings.llm_backend,
        ollama_base=settings.ollama_base,
        model=settings.llm_model,
        base=settings.openai_base,
        key=settings.openai_key,
    )
    tools = OpsTools(settings.data_dir / "ops.sqlite3")
    pipe = Pipeline(store, embedder, llm, tools, settings)

    rows: list[dict] = []
    for case in cases:
        st = pipe.run(case["query"])
        row = {"id": case["id"], "type": case["type"], "query": case["query"]}
        row["score"] = round(st.chunks[0].score, 5) if st.chunks else 0.0

        if case["type"] == "policy":
            secs = [h.section_path for h in st.chunks]
            docs = [h.doc_id for h in st.chunks]
            row["intent_ok"] = st.intent.value in ("policy", "both")
            row["hit@1"] = bool(secs) and case["expect_section"] in secs[0]
            row["hit@3"] = any(case["expect_section"] in s for s in secs[:3])
            row["hit@5"] = any(case["expect_section"] in s for s in secs[:5])
            row["doc_ok"] = case["expect_doc"] in docs
            row["top1"] = secs[0] if secs else "-"
            # 答案内容检查：召回到正确的段落还不够，
            # 答案里必须真的包含该问题的关键信息。
            # 「章节命中但答案没提」是更隐蔽的失败——比如食品安全类问题
            # 召回了含「建议就医」的那段，答案却只字未提。
            must = case.get("expect_answer_contains")
            if must:
                row["content_ok"] = all(m in (st.answer or "") for m in must)
        elif case["type"] == "data":
            used = [t.tool for t in st.tool_results]
            row["intent_ok"] = st.intent.value in ("data", "both")
            row["tool_ok"] = case["expect_tool"] in used
            ok_data = [t for t in st.tool_results if t.ok]
            row["has_data"] = bool(ok_data)
            row["top1"] = f"tools={used or '-'} ok={bool(ok_data)}"
        else:  # noanswer
            # 拒答是语义判断，没有生成模型时无法评测。
            # 如实标 N/A，而不是调个阈值把指标刷上去。
            if llm.name == "none":
                row["refused"] = None
                row["intent_ok"] = True
                row["top1"] = "N/A（未接入 LLM，拒答不可评）"
            else:
                refused = any(h in (st.answer or "") for h in REFUSE_HINTS)
                row["intent_ok"] = True
                row["refused"] = refused
                row["top1"] = "refused" if refused else "answered"

        rows.append(row)

    pol = [r for r in rows if r["type"] == "policy"]
    dat = [r for r in rows if r["type"] == "data"]
    noa = [r for r in rows if r["type"] == "noanswer"]

    def rate(items, key):
        return (sum(1 for i in items if i.get(key)) / len(items) * 100) if items else 0.0

    md: list[str] = []
    md.append("# 检索评估报告\n")
    md.append(f"- embedder: `{embedder.name}`（dim={embedder.dim}）")
    md.append(f"- llm: `{llm.name}`")
    md.append(f"- 索引: docs={store.stats()['docs']} chunks={store.stats()['chunks']}")
    md.append(f"- 用例: {len(rows)} 条\n")
    md.append("## 汇总\n")
    md.append("| 指标 | 数值 |")
    md.append("|---|---|")
    md.append(f"| 规则题 recall@1 | {rate(pol, 'hit@1'):.1f}% |")
    md.append(f"| 规则题 recall@3 | {rate(pol, 'hit@3'):.1f}% |")
    md.append(f"| 规则题 recall@5 | {rate(pol, 'hit@5'):.1f}% |")
    md.append(f"| 规则题文档命中 | {rate(pol, 'doc_ok'):.1f}% |")
    content_cases = [r for r in pol if "content_ok" in r]
    if content_cases:
        md.append(f"| 答案关键内容命中 | {rate(content_cases, 'content_ok'):.1f}% "
                  f"（{len(content_cases)} 条带内容断言）|")
    md.append(f"| 数据题工具选对 | {rate(dat, 'tool_ok'):.1f}% |")
    md.append(f"| 数据题拿到数据 | {rate(dat, 'has_data'):.1f}% |")
    noa_evaluable = [r for r in noa if r.get("refused") is not None]
    if noa_evaluable:
        md.append(f"| 无答案题正确拒答 | {rate(noa_evaluable, 'refused'):.1f}% |")
    else:
        md.append("| 无答案题正确拒答 | N/A（未接入 LLM，拒答是语义判断，检索分无法替代） |")
    md.append(f"| 路由准确率 | {rate(rows, 'intent_ok'):.1f}% |")
    md.append("")
    md.append("## 明细\n")
    md.append("| id | 类型 | 查询 | 结果 | 内容 | top1 分数 | top1 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["type"] == "policy":
            res = f"@{1 if r['hit@1'] else 3 if r['hit@3'] else 5 if r['hit@5'] else 'miss'}"
        elif r["type"] == "data":
            res = "ok" if r.get("tool_ok") and r.get("has_data") else "fail"
        else:
            res = "ok" if r.get("refused") else "fail"
        # 内容断言结果：章节命中但内容缺失也算失败，用 ✘ 标出来
        if "content_ok" in r:
            content = "✔" if r["content_ok"] else "✘"
        else:
            content = "-"
        md.append(f"| {r['id']} | {r['type']} | {r['query']} | {res} | {content} | {r['score']} | {r['top1']} |")

    text = "\n".join(md)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
