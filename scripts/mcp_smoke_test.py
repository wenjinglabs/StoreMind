"""MCP server 协议级冒烟测试。

不是「调用一下工具看有没有报错」，而是按 JSON-RPC 2.0 + MCP 2025-06-18 的握手顺序走：
initialize → notifications/initialized → tools/list → tools/call → resources/read。
顺带验证三类错误码（方法不存在 -32601 / 参数错误 -32602 / JSON 语法错误 -32700），
以及「通知消息不能有响应」这条容易写错的协议细节。

用 `--zero-dep` 可以在没有 ollama、没有生成模型的环境下跑（对应项目的零依赖路径）。

    python scripts/mcp_smoke_test.py
    python scripts/mcp_smoke_test.py --zero-dep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOLS = {
    "search_knowledge", "list_knowledge_docs", "ask_storemind",
    "list_stores", "store_sales", "top_items", "redemption_stats", "inventory_alert",
    "propose_ticket", "propose_restock", "list_pending_actions",
    "confirm_action", "reject_action",
}


class Session:
    """一行一条 JSON 的 stdio 客户端。"""

    def __init__(self, zero_dep: bool) -> None:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if zero_dep:
            env["SM_EMBED_BACKEND"] = "hashing"
            env["SM_LLM_BACKEND"] = "none"
            # 零依赖路径必须连数据库也不依赖：显式钉回 SQLite。
            env["SM_DB_BACKEND"] = "sqlite"
            env["SM_OPS_BACKEND"] = "sqlite"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.mcp.server", "--stdio"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self._id = 0

    def send(self, method: str, params: dict | None = None) -> int:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)
        return self._id

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.send(method, params)
        return self.recv()

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self.call("tools/call", {"name": name, "arguments": arguments or {}})

    def _write(self, msg: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def write_raw(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"服务端没有响应就退出了。stderr:\n{err}")
        return json.loads(line)

    def close(self) -> tuple[int, str]:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        err = self.proc.stderr.read() if self.proc.stderr else ""
        return code, err


def _ok(resp: dict, msg_id: int) -> dict:
    assert resp.get("id") == msg_id, f"响应 id 不匹配：期望 {msg_id}，实际 {resp.get('id')}"
    assert "error" not in resp, f"返回错误：{resp['error']}"
    return resp.get("result") or {}


def _tool_payload(resp: dict, msg_id: int) -> tuple[dict, str, bool]:
    result = _ok(resp, msg_id)
    content = result.get("content") or []
    text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
    return result.get("structuredContent") or {}, text, bool(result.get("isError"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-dep", action="store_true", help="用 hashing + none 后端跑，不需要 ollama")
    parser.add_argument("--out", default="data/mcp_report.txt")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append((name, True, str(detail or "")))
            print(f"[PASS] {name} {detail or ''}")
        except Exception as e:
            checks.append((name, False, str(e)))
            print(f"[FAIL] {name}: {e}")

    s = Session(zero_dep=args.zero_dep)
    try:
        def t_initialize():
            r = _ok(s.call("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "storemind-smoke", "version": "1.0"},
            }), 1)
            assert r.get("protocolVersion") == "2025-06-18", r
            assert r["serverInfo"]["name"] == "storemind", r
            assert "tools" in r["capabilities"] and "resources" in r["capabilities"], r
            return f"protocol={r['protocolVersion']} server={r['serverInfo']['name']}"
        check("initialize 握手", t_initialize)

        # 通知没有 id，不应产生响应。用下一条 ping 的 id 对齐来验证。
        s.notify("notifications/initialized")
        ping_id = s.send("ping")
        check("通知无响应 + ping", lambda: _ok(s.recv(), ping_id) is not None and "pong")

        def t_tools_list():
            r = _ok(s.call("tools/list"), s._id)
            names = {t["name"] for t in r["tools"]}
            assert names == EXPECTED_TOOLS, f"工具集合不一致：{names ^ EXPECTED_TOOLS}"
            for t in r["tools"]:
                assert t["inputSchema"]["type"] == "object", t["name"]
            return f"{len(r['tools'])} 个工具"
        check("tools/list", t_tools_list)

        def t_search():
            resp = s.call_tool("search_knowledge", {"query": "客人说酒有异味怎么办", "top_k": 3})
            payload, text, is_error = _tool_payload(resp, s._id)
            assert not is_error, text
            hits = payload.get("hits") or []
            assert hits, "没有召回任何片段"
            assert any("异味" in (h["section_path"] + h["text"]) for h in hits), \
                [h["section_path"] for h in hits]
            assert text.strip(), "text content 为空"
            return f"{len(hits)} 段，top1={hits[0]['section_path']}"
        check("tools/call search_knowledge", t_search)

        def t_resources_list():
            r = _ok(s.call("resources/list"), s._id)
            assert len(r["resources"]) >= 8, len(r["resources"])
            assert r["resources"][0]["uri"].startswith("storemind://corpus/"), r["resources"][0]
            return f"{len(r['resources'])} 个文档资源"
        check("resources/list", t_resources_list)

        def t_resources_read():
            listed = _ok(s.call("resources/list"), s._id)["resources"]
            r = _ok(s.call("resources/read", {"uri": listed[0]["uri"]}), s._id)
            text = r["contents"][0]["text"]
            assert text.startswith("#"), text[:40]
            return f"{listed[0]['uri']} · {len(text)} 字符"
        check("resources/read", t_resources_read)

        def t_list_stores():
            payload, text, is_error = _tool_payload(s.call_tool("list_stores"), s._id)
            assert not is_error, text
            rows = payload.get("data") or []
            assert len(rows) == 4, rows
            return f"{len(rows)} 家门店"
        check("tools/call list_stores", t_list_stores)

        def t_store_sales():
            payload, text, is_error = _tool_payload(
                s.call_tool("store_sales", {"store_name": "广州天河店", "period": "last_7d"}), s._id
            )
            assert not is_error, text
            data = payload.get("data") or {}
            assert data.get("revenue") is not None, data
            assert data.get("orders"), data
            return f"营业额={data['revenue']} 订单={data['orders']}"
        check("tools/call store_sales", t_store_sales)

        def t_ask():
            payload, text, is_error = _tool_payload(
                s.call_tool("ask_storemind", {"query": "客人说酒有异味怎么办", "thread_id": "smoke"}),
                s._id,
            )
            assert not is_error, text
            assert payload.get("answer", "").strip(), payload
            assert payload.get("intent") in ("policy", "both"), payload.get("intent")
            assert payload.get("citations"), "引用为空"
            assert payload.get("trace"), "执行轨迹为空"
            return f"intent={payload['intent']} 引用={len(payload['citations'])} 轨迹={len(payload['trace'])} 步"
        check("tools/call ask_storemind", t_ask)

        def t_workflow():
            # 客诉工单：提案 → 店员确认
            p, text, err = _tool_payload(s.call_tool("propose_ticket", {
                "store_name": "广州天河店", "summary": "客人反映酒有异味",
                "category": "酒水品质", "role": "clerk", "actor": "smoke",
            }), s._id)
            assert not err, text
            assert p["status"] == "pending" and p["requires_role"] == "clerk", p
            ticket_id = p["action_id"]
            c, text, err = _tool_payload(s.call_tool("confirm_action", {
                "action_id": ticket_id, "role": "clerk", "actor": "smoke",
            }), s._id)
            assert not err and c["status"] == "confirmed", text
            assert c["result"]["ticket_id"].startswith("T-"), c["result"]

            # 补货：店员确认被拒，店长确认成功
            r, text, err = _tool_payload(s.call_tool("propose_restock", {
                "store_name": "广州天河店", "item_name": "莫吉托", "qty": 20,
                "role": "clerk", "actor": "smoke",
            }), s._id)
            assert not err, text
            assert r["requires_role"] == "manager", r
            restock_id = r["action_id"]
            _, text, denied = _tool_payload(s.call_tool("confirm_action", {
                "action_id": restock_id, "role": "clerk",
            }), s._id)
            assert denied and "manager" in text, text
            m, text, err = _tool_payload(s.call_tool("confirm_action", {
                "action_id": restock_id, "role": "manager", "actor": "smoke-manager",
            }), s._id)
            assert not err and m["status"] == "confirmed", text
            return f"ticket={c['result']['ticket_id']} restock={m['result']['request_id']} 越权已拦截"
        check("tools/call 工作流：提案→确认→权限", t_workflow)

        def t_pending():
            p, text, err = _tool_payload(s.call_tool("list_pending_actions", {}), s._id)
            assert not err, text
            assert isinstance(p.get("actions"), list), p
            return f"{len(p['actions'])} 条待办"
        check("tools/call list_pending_actions", t_pending)

        def t_unknown_method():
            resp = s.call("does/not/exist")
            assert resp["error"]["code"] == -32601, resp
            return "code=-32601"
        check("未知方法返回 -32601", t_unknown_method)

        def t_unknown_tool():
            resp = s.call("tools/call", {"name": "no_such_tool", "arguments": {}})
            assert resp["error"]["code"] == -32602, resp
            return "code=-32602"
        check("未知工具返回 -32602", t_unknown_tool)

        def t_bad_json():
            s.write_raw("{这不是 JSON")
            resp = s.recv()
            assert resp["error"]["code"] == -32700, resp
            return "code=-32700"
        check("非法 JSON 返回 -32700", t_bad_json)
    finally:
        code, stderr = s.close()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    lines = [
        f"MCP smoke test · {datetime.now().isoformat(timespec='seconds')}",
        f"零依赖模式: {'是 (hashing + none)' if args.zero_dep else '否 (读 .env)'}",
        f"协议: 2025-06-18 · 传输: stdio",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" · {detail}" if detail else ""))
    lines += ["", f"结果 {passed}/{total}", f"服务端退出码: {code}"]
    if stderr.strip():
        lines += ["", "服务端 stderr（正常日志，不应含 Traceback）:", stderr.strip()]
    report = "\n".join(lines)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\n报告已写入 {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
