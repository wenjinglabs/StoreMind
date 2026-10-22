"""Agent 层 A/B 对照：自检重查到底带来了多少收益。

这是最关键的一个实验。前面的实现如果不能量化收益，那就是自欺欺人。

对照两组，同一批问题：
- A 组：关掉 Agent 层（SM_AGENT_MAX_ATTEMPTS=0），纯流水线
- B 组：开自检重查

量的东西：
1. recall@1 / @3 —— 检索质量有没有提升
2. 触发重查的比例 —— 代价有多大
3. 端到端延迟 —— 每多一次重查要多花多久
4. 「重查后 top1 变了且变得更对」的数量 —— 收益的直接证据

用法：
    python scripts/probe_agent_ab.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import GraphAgent
from app.agent.llm import build_llm
from app.config import settings
from app.rag.embedder import build_embedder
from app.rag.store import build_store
from app.tools.ops_tools import build_ops_tools


def load_cases() -> list[dict]:
    return json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))


def run(agent: GraphAgent, cases: list[dict], tag: str) -> dict:
    hit1 = hit3 = hit5 = total = 0
    n_retry = 0
    latencies: list[int] = []
    changed_better = 0
    changed_worse = 0
    ans_ok = 0          # 答案引用了正确来源
    ans_refused = 0     # 答案拒答了（在没有依据时拒答是对的，但这里只统计数量）
    details: list[str] = []

    for c in cases:
        if c["type"] != "policy":
            continue
        total += 1
        t0 = time.perf_counter()
        st = agent.run(c["query"], thread_id=f"ab-{tag}-{c['id']}")
        latencies.append(int((time.perf_counter() - t0) * 1000))

        secs = [h.section_path for h in st.chunks]
        expect = c["expect_section"]
        h1 = bool(secs) and expect in secs[0]
        h3 = any(expect in s for s in secs[:3])
        h5 = any(expect in s for s in secs[:5])
        hit1 += h1
        hit3 += h3
        hit5 += h5

        # 答案质量：答案引用的编号对应的 chunk，是否落在期望章节里。
        # 这个指标比 recall@1 严——检索对了但模型引了别的段，答案照样错。
        cited_ok = False
        for ci in st.citations:
            if expect in ci.get("section_path", ""):
                cited_ok = True
                break
        ans_ok += cited_ok
        if "现有资料里没有" in (st.answer or ""):
            ans_refused += 1

        retried = any(s.name == "rewrite" for s in st.steps)
        if retried:
            n_retry += 1
            # 第一轮 top1：rewrite 之前的那个 retrieve
            first_top = "-"
            seen = False
            for s in st.steps:
                if s.name == "rewrite":
                    seen = True
                if s.name == "retrieve" and not seen and first_top == "-":
                    d = s.detail
                    if "→" in d:
                        first_top = d.split("→", 1)[1].strip().split(" | ")[0]
            final_top = secs[0] if secs else "-"
            if final_top != first_top:
                if h1 and expect not in first_top:
                    changed_better += 1
                elif not h1 and expect in first_top:
                    changed_worse += 1
            details.append(
                f"   {c['id']} {c['query']}\n"
                f"      第一轮: {first_top}\n"
                f"      最终:   {final_top}  检索{'✔' if h1 else '✘'} 引用{'✔' if cited_ok else '✘'}"
            )

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    return {
        "tag": tag,
        "n": total,
        "hit1": hit1 / total * 100 if total else 0,
        "hit3": hit3 / total * 100 if total else 0,
        "hit5": hit5 / total * 100 if total else 0,
        "ans_ok": ans_ok / total * 100 if total else 0,
        "ans_refused": ans_refused,
        "n_retry": n_retry,
        "avg_latency_ms": int(avg_lat),
        "max_latency_ms": max(latencies) if latencies else 0,
        "changed_better": changed_better,
        "changed_worse": changed_worse,
        "details": details,
    }


def main() -> None:
    settings.ensure_dirs()
    store = build_store(settings)
    emb = build_embedder(settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model)
    llm = build_llm(settings.llm_backend, ollama_base=settings.ollama_base,
                    model=settings.llm_model, base=settings.openai_base,
                    key=settings.openai_key, timeout=settings.llm_timeout)
    tools = build_ops_tools(settings)
    cases = load_cases()

    out: list[str] = []
    out.append("=" * 78)
    out.append("Agent 层 A/B 对照：自检重查的收益")
    out.append("=" * 78)

    # A 组：关掉 Agent 层
    settings.agent_max_attempts = 0
    agent_a = GraphAgent(store, emb, llm, tools, settings)
    a = run(agent_a, cases, "off")
    agent_a.close()

    # B 组：开自检重查
    settings.agent_max_attempts = 2
    agent_b = GraphAgent(store, emb, llm, tools, settings)
    b = run(agent_b, cases, "on")
    agent_b.close()

    out.append("")
    out.append("| 指标 | A 关掉 Agent | B 开启自检重查 | 变化 |")
    out.append("|---|---|---|---|")
    out.append(f"| recall@1 | {a['hit1']:.1f}% | {b['hit1']:.1f}% | {b['hit1']-a['hit1']:+.1f} |")
    out.append(f"| recall@3 | {a['hit3']:.1f}% | {b['hit3']:.1f}% | {b['hit3']-a['hit3']:+.1f} |")
    out.append(f"| recall@5 | {a['hit5']:.1f}% | {b['hit5']:.1f}% | {b['hit5']-a['hit5']:+.1f} |")
    out.append(f"| **答案引用正确来源** | {a['ans_ok']:.1f}% | {b['ans_ok']:.1f}% | {b['ans_ok']-a['ans_ok']:+.1f} |")
    out.append(f"| 答案拒答数 | {a['ans_refused']} | {b['ans_refused']} | {b['ans_refused']-a['ans_refused']:+d} |")
    out.append(f"| 平均延迟 | {a['avg_latency_ms']}ms | {b['avg_latency_ms']}ms | {b['avg_latency_ms']-a['avg_latency_ms']:+d} |")
    out.append(f"| 最大延迟 | {a['max_latency_ms']}ms | {b['max_latency_ms']}ms | {b['max_latency_ms']-a['max_latency_ms']:+d} |")
    out.append(f"| 触发重查 | {a['n_retry']}/{a['n']} | {b['n_retry']}/{b['n']} | {b['n_retry']-a['n_retry']:+d} |")
    out.append(f"| 重查后 top1 变对 | {a['changed_better']} | {b['changed_better']} | {b['changed_better']-a['changed_better']:+d} |")
    out.append(f"| 重查后 top1 变差 | {a['changed_worse']} | {b['changed_worse']} | {b['changed_worse']-a['changed_worse']:+d} |")
    out.append("")
    out.append("B 组触发重查的用例明细：")
    for d in b["details"]:
        out.append(d)

    text = "\n".join(out)
    Path("data/agent_ab.txt").write_text(text, encoding="utf-8")
    print("written data/agent_ab.txt")


if __name__ == "__main__":
    main()
