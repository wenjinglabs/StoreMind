"""自实现的 MCP server。

**为什么不用官方 SDK**：项目要证明的是「知道协议长什么样」，不是「会装包」。
MCP 的传输层是 JSON-RPC 2.0，核心只有三件事：
1. `initialize` 握手，协商协议版本和能力；
2. `tools/list` 报工具清单，`tools/call` 执行；
3. `resources/list` / `resources/read` 暴露可读资源。

这份实现只用标准库（json / sys / http.server），不依赖 `mcp` 包。
stdio 是 MCP 的标准传输：每行一条 JSON，服务端 stdout 只允许出现协议消息，
所以日志一律走 stderr——这一条踩过就知道，混一行 print 到 stdout 客户端直接解析失败。

协议版本跟进 `2025-06-18`：新增的 `structuredContent` 也实现了，
工具返回既给人读的 text，也给客户端机器可读的结构化结果。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..agent.graph import GraphAgent
from ..agent.llm import build_llm
from ..config import Settings, settings
from ..rag.embedder import build_embedder
from ..rag.retriever import hybrid_search
from ..rag.store import Store
from ..tools.ops_tools import OpsTools, ToolResult

SERVER_NAME = "storemind"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"

PERIODS = {"today", "yesterday", "last_7d", "last_30d", "this_month"}

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolError(Exception):
    """工具执行失败。映射成 tools/call 的 isError，而不是 JSON-RPC 错误。

    协议上这两个不一样：客户端传了不存在的工具名是 -32602（调用方写错了），
    工具跑了但查不到数据是 isError=true（业务结果，模型可以读着调整）。
    """


def _text_result(text: str, structured: dict | None = None, is_error: bool = False) -> dict:
    result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _period(args: dict) -> str:
    period = str(args.get("period") or "last_7d")
    if period not in PERIODS:
        raise ToolError(f"period 只能是 {sorted(PERIODS)} 之一，收到 {period!r}")
    return period


def _store_name(args: dict) -> str:
    name = str(args.get("store_name") or "").strip()
    if not name:
        raise ToolError("缺少 store_name，例如「广州天河店」")
    return name


class StoremindTools:
    """把现有能力包成 MCP 工具。

    刻意分两层：
    - **原子工具**（检索 / 5 个经营数据查询）：客户端自己的模型可以自由组合；
    - **ask_storemind**：直接复用项目里的 Agent（路由 + 自检重查 + 引用），
      适合不想在客户端重写编排的场景。

    前者证明「工具是标准的、可被任意 MCP 客户端消费」，
    后者保证「换了客户端，项目最有价值的那层编排还在」。
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.s = cfg or settings
        self.s.ensure_dirs()
        self.store = Store(self.s.db_path)
        self.embedder = build_embedder(
            self.s.embed_backend, ollama_base=self.s.ollama_base, model=self.s.embed_model
        )
        self.tools = OpsTools(self.s.data_dir / "ops.sqlite3")
        self._agent: GraphAgent | None = None

    # Agent 延迟构建：只调 search_knowledge 的客户端不需要加载 LangGraph 和模型。
    @property
    def agent(self) -> GraphAgent:
        if self._agent is None:
            llm = build_llm(
                self.s.llm_backend,
                ollama_base=self.s.ollama_base,
                model=self.s.openai_model
                if self.s.llm_backend in ("openai", "deepseek")
                else self.s.llm_model,
                base=self.s.openai_base,
                key=self.s.openai_key,
                timeout=self.s.llm_timeout,
            )
            self._agent = GraphAgent(self.store, self.embedder, llm, self.tools, self.s)
        return self._agent

    # ---------- 工具实现 ----------

    def search_knowledge(self, args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("缺少 query")
        top_k = int(args.get("top_k") or self.s.final_top_k)
        top_k = max(1, min(top_k, 20))
        hits = hybrid_search(
            self.store, self.embedder, query,
            top_k=top_k, vector_k=self.s.vector_top_k,
            bm25_k=self.s.bm25_top_k, rrf_k=self.s.rrf_k,
        )
        payload = {
            "query": query,
            "hits": [
                {
                    "doc_id": h.doc_id,
                    "doc_title": h.doc_title,
                    "section_path": h.section_path,
                    "score": round(h.score, 5),
                    "text": h.text,
                }
                for h in hits
            ],
        }
        if not hits:
            return _text_result("知识库里没有召回任何片段。", payload)
        lines = [
            f"[{i}] 《{h.doc_title} · {h.section_path}》 score={h.score:.5f}\n{h.text}"
            for i, h in enumerate(hits, start=1)
        ]
        return _text_result("\n\n".join(lines), payload)

    def list_knowledge_docs(self, args: dict) -> dict:
        docs = self.store.list_docs()
        lines = [f"- {d['doc_id']} · {d['title']}（{d['chunk_count']} 段）" for d in docs]
        return _text_result("\n".join(lines), {"docs": docs})

    def list_stores(self, args: dict) -> dict:
        return self._ops("list_stores", self.tools.list_stores())

    def store_sales(self, args: dict) -> dict:
        return self._ops(
            "store_sales",
            self.tools.store_sales(_store_name(args), _period(args)),  # type: ignore[arg-type]
        )

    def top_items(self, args: dict) -> dict:
        limit = int(args.get("limit") or 5)
        limit = max(1, min(limit, 20))
        return self._ops(
            "top_items",
            self.tools.top_items(_store_name(args), _period(args), limit),  # type: ignore[arg-type]
        )

    def redemption_stats(self, args: dict) -> dict:
        return self._ops(
            "redemption_stats",
            self.tools.redemption_stats(_store_name(args), _period(args)),  # type: ignore[arg-type]
        )

    def inventory_alert(self, args: dict) -> dict:
        threshold = int(args.get("threshold") or 10)
        return self._ops(
            "inventory_alert",
            self.tools.inventory_alert(_store_name(args), threshold),
        )

    def ask_storemind(self, args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("缺少 query")
        thread_id = str(args.get("thread_id") or "mcp")
        st = self.agent.run(query, thread_id=thread_id)
        payload = {
            "query": st.query,
            "thread_id": thread_id,
            "intent": st.intent.value,
            "route_reason": st.route_reason,
            "confidence": round(st.confidence, 2),
            "answer": st.answer,
            "judge_ok": st.judge_ok,
            "judge_reason": st.judge_reason,
            "retries": sum(1 for s in st.steps if s.name == "rewrite"),
            "degraded": st.degraded,
            "citations": st.citations,
            "data": [
                {"tool": t.tool, "ok": t.ok, "note": t.note, "data": t.data}
                for t in st.tool_results
            ],
            "trace": st.trace(),
        }
        return _text_result(st.answer, payload)

    @staticmethod
    def _ops(tool: str, result: ToolResult) -> dict:
        payload = {
            "tool": result.tool,
            "ok": result.ok,
            "note": result.note,
            "data": result.data,
        }
        if not result.ok:
            return _text_result(f"{tool} 未查到数据：{result.note}", payload, is_error=True)
        if isinstance(result.data, list):
            text = "\n".join(
                f"[{i}] " + "，".join(f"{k}={v}" for k, v in row.items())
                if isinstance(row, dict) else f"[{i}] {row}"
                for i, row in enumerate(result.data, start=1)
            )
        elif isinstance(result.data, dict):
            text = "，".join(f"{k}={v}" for k, v in result.data.items())
        else:
            text = str(result.data)
        return _text_result(f"{text}\n（{result.note}）" if result.note else text, payload)

    # ---------- 资源 ----------

    def resource_list(self) -> list[dict]:
        """语料文档作为 MCP resources 暴露。

        工具是「模型决定要不要调用」，资源是「客户端可以直接挂进上下文」。
        两种暴露方式对应两种使用方式，不是重复。
        """
        return [
            {
                "uri": f"storemind://corpus/{d['doc_id']}",
                "name": d["title"],
                "description": f"{d['doc_id']} · {d['chunk_count']} 个切片",
                "mimeType": "text/markdown",
            }
            for d in self.store.list_docs()
        ]

    def resource_read(self, uri: str) -> dict:
        prefix = "storemind://corpus/"
        if not uri.startswith(prefix):
            raise RpcError(JSONRPC_INVALID_PARAMS, f"不支持的资源 URI: {uri}")
        doc_id = uri[len(prefix):]
        known = {d["doc_id"] for d in self.store.list_docs()}
        if doc_id not in known:
            raise RpcError(JSONRPC_INVALID_PARAMS, f"资源不存在: {uri}")
        base = self.s.corpus_dir.resolve()
        path = (base / doc_id).resolve()
        # 路径穿越防护：doc_id 来自数据库，但仍然不信任拼接结果。
        if not path.is_relative_to(base) or not path.is_file():
            raise RpcError(JSONRPC_INVALID_PARAMS, f"资源文件不可读: {uri}")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": path.read_text(encoding="utf-8"),
                }
            ]
        }


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "门店运营相关的问题，中文原句即可"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
    },
    "required": ["query"],
}

PERIOD_SCHEMA = {
    "type": "string",
    "enum": sorted(PERIODS),
    "default": "last_7d",
    "description": "统计周期：today / yesterday / last_7d / last_30d / this_month",
}

STORE_SCHEMA = {
    "type": "string",
    "description": "门店名称，支持全称或简称，例如 广州天河店 / 天河店",
}

TOOLS: list[dict] = [
    {
        "name": "search_knowledge",
        "description": "检索门店运营知识库（SOP、制度、产品知识），返回带章节路径和分值的原文片段。适合回答规则、流程、话术类问题。",
        "inputSchema": SEARCH_SCHEMA,
    },
    {
        "name": "list_knowledge_docs",
        "description": "列出知识库里的全部文档及其切片数。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_storemind",
        "description": "把问题交给 StoreMind Agent 完整处理：意图路由、知识库检索/经营数据查询、证据自检与改写重查，返回答案、引用和执行轨迹。适合需要完整回答而不是原始片段的场景。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户原话"},
                "thread_id": {
                    "type": "string",
                    "default": "mcp",
                    "description": "会话 ID。同一 ID 的多次调用共享上下文，用于「那退卡呢」这类追问。",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_stores",
        "description": "列出经营库里全部门店（store_id、名称、城市）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "store_sales",
        "description": "查询某门店在指定周期的订单数、营业额、客单价。只统计已支付订单。",
        "inputSchema": {
            "type": "object",
            "properties": {"store_name": STORE_SCHEMA, "period": PERIOD_SCHEMA},
            "required": ["store_name"],
        },
    },
    {
        "name": "top_items",
        "description": "查询某门店在指定周期的热销品项排行（按营业额降序）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_name": STORE_SCHEMA,
                "period": PERIOD_SCHEMA,
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["store_name"],
        },
    },
    {
        "name": "redemption_stats",
        "description": "查询某门店在指定周期的活动核销次数与金额。",
        "inputSchema": {
            "type": "object",
            "properties": {"store_name": STORE_SCHEMA, "period": PERIOD_SCHEMA},
            "required": ["store_name"],
        },
    },
    {
        "name": "inventory_alert",
        "description": "查询某门店库存低于阈值的品项，按数量升序。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "store_name": STORE_SCHEMA,
                "threshold": {"type": "integer", "minimum": 0, "default": 10},
            },
            "required": ["store_name"],
        },
    },
]

TOOL_METHODS: dict[str, Callable[[dict], dict]] = {
    "search_knowledge": StoremindTools.search_knowledge,
    "list_knowledge_docs": StoremindTools.list_knowledge_docs,
    "ask_storemind": StoremindTools.ask_storemind,
    "list_stores": StoremindTools.list_stores,
    "store_sales": StoremindTools.store_sales,
    "top_items": StoremindTools.top_items,
    "redemption_stats": StoremindTools.redemption_stats,
    "inventory_alert": StoremindTools.inventory_alert,
}


class McpServer:
    def __init__(self, tools: StoremindTools) -> None:
        self.tools = tools
        self.session_id = uuid.uuid4().hex
        self.initialized = False

    # ---------- JSON-RPC ----------

    def handle_message(self, msg: Any) -> dict | list | None:
        if isinstance(msg, list):
            if not msg:
                return self._error(None, JSONRPC_INVALID_REQUEST, "空的批量请求")
            out = [r for r in (self._handle_one(m) for m in msg) if r is not None]
            return out or None
        return self._handle_one(msg)

    def _handle_one(self, msg: Any) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._error(
                msg.get("id") if isinstance(msg, dict) else None,
                JSONRPC_INVALID_REQUEST,
                "不是合法的 JSON-RPC 2.0 消息",
            )

        method = msg.get("method")
        is_notification = "id" not in msg
        if not isinstance(method, str) or not method:
            if is_notification:
                return None
            return self._error(msg.get("id"), JSONRPC_INVALID_REQUEST, "缺少 method")

        try:
            result = self._dispatch(method, msg.get("params") or {})
            return None if is_notification else {"jsonrpc": "2.0", "id": msg.get("id"), "result": result}
        except RpcError as e:
            return None if is_notification else self._error(msg.get("id"), e.code, e.message)
        except Exception as e:  # 协议层绝不能因为一个坏请求挂掉
            traceback.print_exc(file=sys.stderr)
            return None if is_notification else self._error(
                msg.get("id"), JSONRPC_INTERNAL_ERROR, f"服务端异常: {e}"
            )

    def _dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            client_version = params.get("protocolVersion")
            self.initialized = True
            return {
                "protocolVersion": client_version if client_version == PROTOCOL_VERSION else PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "StoreMind 是连锁酒吧/餐饮门店的运营知识库。"
                    "规则类问题用 search_knowledge 或 ask_storemind；"
                    "经营数据用 store_sales / top_items / redemption_stats / inventory_alert。"
                ),
            }

        if method in ("notifications/initialized", "notifications/cancelled"):
            return {}

        if method == "ping":
            return {}

        if method == "tools/list":
            return {"tools": TOOLS}

        if method == "tools/call":
            if not isinstance(params, dict):
                raise RpcError(JSONRPC_INVALID_PARAMS, "params 必须是对象")
            name = params.get("name")
            if not isinstance(name, str) or name not in TOOL_METHODS:
                raise RpcError(JSONRPC_INVALID_PARAMS, f"未知工具: {name}")
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                raise RpcError(JSONRPC_INVALID_PARAMS, "arguments 必须是对象")
            handler = TOOL_METHODS[name]
            try:
                return handler(self.tools, args)
            except ToolError as e:
                return _text_result(f"参数错误: {e}", is_error=True)
            except FileNotFoundError as e:
                return _text_result(f"数据未就绪: {e}", is_error=True)
            except RpcError:
                raise
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                return _text_result(f"工具执行失败: {e}", is_error=True)

        if method == "resources/list":
            return {"resources": self.tools.resource_list()}

        if method == "resources/read":
            uri = params.get("uri") if isinstance(params, dict) else None
            if not isinstance(uri, str):
                raise RpcError(JSONRPC_INVALID_PARAMS, "resources/read 需要 uri")
            return self.tools.resource_read(uri)

        raise RpcError(JSONRPC_METHOD_NOT_FOUND, f"未实现的方法: {method}")

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ---------- 传输 ----------

def serve_stdio(server: McpServer) -> None:
    """stdio 传输：一行一条 JSON。

    标准输出只写协议消息，日志走 stderr。客户端是逐行解析的，
    stdout 混进一行 print 就会让整个会话解析失败。
    """
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    for raw in stdin:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            payload = McpServer._error(None, JSONRPC_PARSE_ERROR, f"JSON 解析失败: {e.msg}")
        else:
            payload = server.handle_message(msg)
        if payload is None:
            continue
        out = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        stdout.write(out.encode("utf-8") + b"\n")
        stdout.flush()


def serve_http(server: McpServer, host: str, port: int) -> None:
    """HTTP 传输：POST /mcp 走 JSON-RPC，GET /health 探活。

    MCP 的 Streamable HTTP 允许服务端直接回 application/json（不必 SSE），
    这里就是那条最简路径。通知类消息回 202，请求类回 200 + JSON。
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = f"StoreMindMCP/{SERVER_VERSION}"

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
            if self.path.rstrip("/") not in ("/mcp", ""):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                msg = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self._send(200, McpServer._error(None, JSONRPC_PARSE_ERROR, f"JSON 解析失败: {e}"))
                return
            payload = server.handle_message(msg)
            if payload is None:
                self.send_response(202)
                self.send_header("Mcp-Session-Id", server.session_id)
                self.end_headers()
                return
            self._send(200, payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send(200, {
                    "ok": True,
                    "server": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "protocol": PROTOCOL_VERSION,
                    "tools": len(TOOLS),
                    "docs": len(server.tools.store.list_docs()),
                })
                return
            self._send(405, {"error": "use POST /mcp"})

        def _send(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Mcp-Session-Id", server.session_id)
            self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[mcp-http] {self.address_string()} {fmt % args}", file=sys.stderr)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[mcp] HTTP listening on http://{host}:{port}/mcp", file=sys.stderr)
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StoreMind MCP server（自实现，无 SDK 依赖）")
    parser.add_argument("--stdio", action="store_true", help="stdio 传输（默认）")
    parser.add_argument("--http", nargs="?", const=8765, type=int, metavar="PORT",
                        help="HTTP 传输，可指定端口（默认 8765）")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址")
    args = parser.parse_args(argv)

    tools = StoremindTools()
    server = McpServer(tools)

    if args.http is not None:
        serve_http(server, args.host, args.http)
    else:
        serve_stdio(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
