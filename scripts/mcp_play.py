"""MCP 协议学习工具：逐帧发请求，把原始 JSON 打印出来。

不是测试（测试看的是断言结果），这里看的是**协议长什么样**：
每一步先解释「这一步在协议里干什么」，再发真实请求，再打印原始响应。

    python scripts/mcp_play.py            # 自动连播，每步停 1 秒
    python scripts/mcp_play.py --step      # 每步回车继续，适合边看边想
    python scripts/mcp_play.py --full      # 不截断工具返回的正文

默认走无模型模式（hashing + none），仍然需要 PostgreSQL（compose 的 5433）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 每步：标题、讲解、要发的原始文本（None 表示由调用方自己构造）
STEPS = [
    (
        "1. initialize：握手，协商版本与能力",
        "客户端第一句话。三个字段缺一不可：protocolVersion（客户端支持的版本）、\n"
        "capabilities（客户端自己有什么能力）、clientInfo（名字和版本，给服务端记日志用）。\n"
        "服务端回 protocolVersion + capabilities + serverInfo，两边取交集才开始工作。",
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "mcp-play", "version": "1.0"}}},
    ),
    (
        "2. notifications/initialized：通知，没有 id",
        "握手完客户端发一条「我准备好了」。注意它**没有 id**——JSON-RPC 里没 id 就是通知，\n"
        "服务端必须不做任何响应。写错这里最常见：多发一条响应，客户端解析就错位了。",
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ),
    (
        "3. ping：验证上一步真的没回响应",
        "发一个带宽度的 ping。如果第 2 步错误地回了响应，这里读到的 id 就不是 3，\n"
        "而是那条通知的响应——这就是冒烟测试里验证「通知无响应」的办法。",
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ),
    (
        "4. tools/list：报工具清单",
        "服务端返回每个工具的四件事：name（唯一标识）、description（给模型看的自然语言说明）、\n"
        "inputSchema（JSON Schema，模型据此填参数）、annotations（可选，只读/破坏性提示）。\n"
        "注意：description 是写给**模型**的，不是写给人的——它决定模型会不会选这个工具。",
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list"},
    ),
    (
        "5. tools/call：执行工具",
        "params 固定两个字段：name（必须和 tools/list 里的一模一样）+ arguments（对象）。\n"
        "返回值不是裸数据：content 是给人/模型读的文本，structuredContent 是给客户端程序读的结构。\n"
        "structuredContent 是 2025-06-18 这一版新增的，旧版本只有 content。\n"
        "注意 isError=false——协议层没出错。",
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "search_knowledge",
                    "arguments": {"query": "客人说酒有异味怎么办", "top_k": 2}}},
    ),
    (
        "6. 工具不存在：-32602，不是 -32601",
        "这里最容易讲错。方法（tools/call）存在，但参数里的 name 不存在——\n"
        "协议层看是「参数错了」，所以是 INVALID_PARAMS(-32602)。\n"
        "只有请求的 method 本身不存在（比如 foo/bar）才是 METHOD_NOT_FOUND(-32601)。",
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "no_such_tool", "arguments": {}}},
    ),
    (
        "7. resources/list：可读资源清单",
        "工具是「模型决定要不要调」，资源是「客户端可以直接挂进上下文」。\n"
        "资源用 URI 标识（这里是 storemind://corpus/<doc_id>），带 mimeType，\n"
        "客户端可以提前读、可以订阅变更（capabilities.resources.subscribe，你这里声明为 false）。",
        {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
    ),
    (
        "8. resources/read：按 URI 读内容",
        "返回 contents 数组，每项 {uri, mimeType, text}。\n"
        "URI 来自上一步 resources/list，不能自己拼错的。",
        {"jsonrpc": "2.0", "id": 8, "method": "resources/read",
         "params": {"uri": "storemind://corpus/product/01-酒水知识与出品标准.md"}},
    ),
    (
        "9. 资源不存在：还是 -32602",
        "错在参数（uri 无效），不是方法。注意这里没有返回 200 + 空内容——\n"
        "「找不到」是错误，不是「找到了但没东西」。",
        {"jsonrpc": "2.0", "id": 9, "method": "resources/read",
         "params": {"uri": "storemind://corpus/不存在.md"}},
    ),
    (
        "10. 未知方法：-32601",
        "method 本身不存在，客户端调错了接口。和第 6 步的区别是：错在「方法名」还是「参数」。",
        {"jsonrpc": "2.0", "id": 10, "method": "does/not/exist"},
    ),
    (
        "11. 非法 JSON：-32700",
        "传输层就解析失败了，连 id 都没有，所以响应里的 id 是 null。\n"
        "这一条证明服务端不能因为一个坏包就崩溃。",
        "{这不是 JSON",
    ),
    (
        "12. 批量请求：数组（2025-06-18 已从规范里移除）",
        "JSON-RPC 2.0 本身允许批量，但 MCP 2025-06-18 明确移除了批量支持。\n"
        "这里保留数组处理是为了兼容老客户端，不算这一版的能力——被问到时这么答才准确。\n"
        "空数组应该回 INVALID_REQUEST(-32600)。",
        [{"jsonrpc": "2.0", "id": 11, "method": "ping"},
         {"jsonrpc": "2.0", "id": 12, "method": "resources/read",
          "params": {"uri": "storemind://corpus/sop/03-客诉处理手册.md"}}],
    ),
]

# 响应里这些键的文本太长了，默认截断；--full 不截
TRIM_KEYS = {"text"}


def _trim(obj, full: bool):
    if full:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in TRIM_KEYS and isinstance(v, str) and len(v) > 220:
                out[k] = v[:220] + f"…（共 {len(v)} 字符，--full 看全文）"
            else:
                out[k] = _trim(v, full)
        return out
    if isinstance(obj, list):
        return [_trim(x, full) for x in obj]
    return obj


class Server:
    def __init__(self) -> None:
        env = os.environ.copy()
        env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "SM_EMBED_BACKEND": "hashing",
            "SM_LLM_BACKEND": "none",
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.mcp.server", "--stdio"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, env=env,
        )

    def send(self, payload) -> None:
        assert self.proc.stdin is not None
        raw = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        self.proc.stdin.write(raw + "\n")
        self.proc.stdin.flush()

    def recv(self):
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"服务端退出，stderr:\n{err}")
        return json.loads(line)

    def close(self) -> str:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.stderr.read() if self.proc.stderr else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", action="store_true", help="每步等回车")
    parser.add_argument("--full", action="store_true", help="不截断正文")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    srv = Server()
    try:
        for title, why, payload in STEPS:
            print("\n" + "=" * 78)
            print(title)
            print("-" * 78)
            print(why)
            print()
            if args.step:
                input("回车发送 → ")
            else:
                time.sleep(0.8)

            srv.send(payload)
            print("→ " + (payload if isinstance(payload, str)
                          else json.dumps(payload, ensure_ascii=False)))

            # 通知没有响应，不能读；只有带 id 的才等回包
            is_notification = isinstance(payload, dict) and "id" not in payload
            if is_notification:
                print("← （通知，服务端不应有任何响应）")
                continue

            resp = srv.recv()
            print("← " + json.dumps(_trim(resp, args.full), ensure_ascii=False,
                                    indent=2).replace("\n", "\n  "))
            if args.step:
                input()
        if not args.step:
            time.sleep(0.3)
    finally:
        stderr = srv.close()

    print("\n" + "=" * 78)
    print("服务端 stderr（只应是日志，出现 Traceback 就是代码有问题）：")
    print(stderr.strip() or "（空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
