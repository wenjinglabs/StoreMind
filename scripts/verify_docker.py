"""Docker 部署验证：对容器里跑起来的两个入口做真实请求。

验证的不是「容器启动了」，而是「零依赖配置下网页、/chat、MCP 都能用」：
1. app /health 必须是 hashing + none（不依赖宿主机模型）
2. 规则题走知识库并带引用
3. 数据题走白名单工具并拿到真实聚合
4. 前端页面能打开
5. MCP /health、initialize、search_knowledge、ask_storemind 都能响应

    python scripts/verify_docker.py
    python scripts/verify_docker.py --app http://127.0.0.1:8000 --mcp http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="http://127.0.0.1:8000")
    ap.add_argument("--mcp", default="http://127.0.0.1:8765")
    ap.add_argument("--expect-embed", default="hashing", help="hashing | ollama")
    ap.add_argument("--expect-llm", default="none", help="none | ollama | openai")
    ap.add_argument("--out", default="data/docker_verify.txt")
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            ms = int((time.perf_counter() - t0) * 1000)
            checks.append((name, True, f"{detail} ({ms}ms)"))
            print(f"[PASS] {name} {detail} ({ms}ms)")
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            checks.append((name, False, f"{e} ({ms}ms)"))
            print(f"[FAIL] {name}: {e}")

    c = httpx.Client(timeout=60.0)

    def t_health():
        h = c.get(f"{args.app}/health").json()
        assert h["ok"] is True, h
        assert h["embed_backend"] == args.expect_embed, h
        assert h["llm_backend"] == args.expect_llm, h
        assert h["stats"]["docs"] == 8 and h["stats"]["chunks"] == 45, h["stats"]
        return f"embed={h['embed_backend']} llm={h['llm_backend']} docs={h['stats']['docs']} chunks={h['stats']['chunks']}"

    def t_chat_policy():
        d = c.post(f"{args.app}/chat", json={"query": "会员卡能跨店用吗", "thread_id": "docker-policy"}).json()
        assert d["intent"] == "policy", d["intent"]
        assert d["answer"].strip(), "答案为空"
        assert d["citations"], "引用为空"
        return f"intent={d['intent']} 引用={len(d['citations'])} 答案={len(d['answer'])}字"

    def t_chat_data():
        d = c.post(f"{args.app}/chat", json={"query": "上周广州天河店卖了多少", "thread_id": "docker-data"}).json()
        assert d["intent"] == "data", d["intent"]
        assert d["data"] and d["data"][0]["tool"] == "store_sales", d.get("data")
        assert d["data"][0]["ok"] is True, d["data"]
        revenue = d["data"][0]["data"]["revenue"]
        assert revenue > 0, revenue
        return f"tool=store_sales revenue={revenue}"

    def t_frontend():
        r = c.get(f"{args.app}/")
        assert r.status_code == 200, r.status_code
        assert "StoreMind" in r.text, "页面内容不含 StoreMind"
        return f"HTTP {r.status_code} · {len(r.text)} 字符"

    def t_mcp_health():
        h = c.get(f"{args.mcp}/health").json()
        assert h["ok"] is True, h
        assert h["tools"] == 8 and h["docs"] == 8, h
        return f"tools={h['tools']} docs={h['docs']}"

    def t_mcp_initialize():
        r = c.post(f"{args.mcp}/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }).json()
        assert r["result"]["protocolVersion"] == "2025-06-18", r
        assert r["result"]["serverInfo"]["name"] == "storemind", r
        return f"protocol={r['result']['protocolVersion']}"

    def t_mcp_search():
        r = c.post(f"{args.mcp}/mcp", json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "search_knowledge", "arguments": {"query": "客人说酒有异味怎么办"}},
        }).json()
        hits = r["result"]["structuredContent"]["hits"]
        assert hits, "MCP 检索为空"
        return f"{len(hits)} 段 · top1={hits[0]['section_path']}"

    def t_mcp_ask():
        r = c.post(f"{args.mcp}/mcp", json={
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "ask_storemind", "arguments": {"query": "上周广州天河店卖了多少", "thread_id": "docker-mcp"}},
        }).json()
        payload = r["result"]["structuredContent"]
        assert payload["answer"].strip(), payload
        assert payload["data"] and payload["data"][0]["ok"] is True, payload.get("data")
        return f"intent={payload['intent']} 数据={payload['data'][0]['tool']} 轨迹={len(payload['trace'])}步"

    check("app /health（零依赖配置）", t_health)
    check("app /chat 规则题", t_chat_policy)
    check("app /chat 数据题", t_chat_data)
    check("app 前端页面", t_frontend)
    check("mcp /health", t_mcp_health)
    check("mcp initialize", t_mcp_initialize)
    check("mcp tools/call search_knowledge", t_mcp_search)
    check("mcp tools/call ask_storemind", t_mcp_ask)

    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [
        f"Docker 部署验证 · {datetime.now().isoformat(timespec='seconds')}",
        f"app={args.app}  mcp={args.mcp}",
        f"期望后端: embed={args.expect_embed} llm={args.expect_llm}",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name} · {detail}")
    lines += ["", f"结果 {passed}/{len(checks)}"]
    report = "\n".join(lines)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\n报告已写入 {out}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
