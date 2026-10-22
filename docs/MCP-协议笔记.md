# MCP 协议笔记（对着 StoreMind 的代码学）

> 目标：被问到时能从「协议长什么样」讲到「你为什么这么写」，而不是背名词。
> 配套工具：`scripts/mcp_play.py`（逐帧打印真实请求/响应）、`scripts/mcp_smoke_test.py`（断言版）。

---

## 0. 30 秒回答（背下来）

> MCP 是让 AI 应用发现并调用外部能力的协议，传输层就是 JSON-RPC 2.0。
> 流程是 `initialize` 握手协商版本和能力，然后 `tools/list` 报清单、`tools/call` 执行，
> 或者 `resources/list` / `resources/read` 读资源。
> 我这份是用标准库自己实现的，没用官方 SDK，所以协议细节都踩过：
> 通知不能回响应、方法不存在是 -32601 而参数错是 -32602、
> 工具跑失败要用 `isError` 而不是 JSON-RPC 错误。

---

## 1. 三个角色

```
Host（Claude Desktop / Cursor / 你的 Agent 应用）
  └─ Client ──(JSON-RPC over stdio | Streamable HTTP)── Server（StoreMind）
```

- **Host**：用户用的那个应用，负责界面、权限、把模型和客户端接起来。
- **Client**：协议会话的一端，由 Host 创建，一个 Server 对应一个 Client 连接。
- **Server**：暴露 tools / resources / prompts 的那一方，这里就是 StoreMind。

**常见误解**：MCP 不是「让模型上网的工具」，也不是模型的一部分。它不规定模型怎么想，
只规定「能力怎么被发现、怎么被调用、失败怎么表达」。

---

## 2. 一次会话的完整时序

```
Client                                    Server
  │ initialize(protocolVersion, caps) ────→ │
  │ ←──── result(protocolVersion, caps)     │
  │ notifications/initialized ────────────→ │  （通知，无 id，不回应）
  │ tools/list ───────────────────────────→ │
  │ ←──── {tools: [{name, description, inputSchema}]}
  │ tools/call(name, arguments) ──────────→ │
  │ ←──── {content: [...], isError: false, structuredContent: {...}}
```

四条规矩：

1. **必须先 initialize**，之后才能调工具。新版规范把这条从 SHOULD 改成了 MUST。
2. **版本取交集**。客户端发 2025-06-18，服务端不支持就回自己支持的版本，客户端决定继续还是断开。
3. **能力要对得上**。服务端声明的 `capabilities.tools` / `capabilities.resources`
   才代表它支持对应方法；客户端声明的 `roots` / `sampling` / `elicitation` 同理。
4. **【无 id 就是通知】** 通知没有响应，一条都不能回。

---

## 3. 七条必须能讲对的规则

### 3.1 没有 id 就是通知，不能回响应

```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}     ← 通知
{"jsonrpc": "2.0", "id": 3, "method": "ping"}                 ← 请求
```

代码：`McpServer._handle_one()` 里 `is_notification = "id" not in msg`，
所有返回路径都写成 `return None if is_notification else {...}`。

**为什么严重**：客户端是逐行读响应、按 id 配对的。多回一条，后面所有响应都错位一格，
表现为「工具调用返回了上一个请求的结果」这种极难排查的 bug。

**怎么验证**：发完通知立刻发 ping，检查收到的 id 是不是 ping 的 id
（`mcp_smoke_test.py` 的「通知无响应 + ping」这一条）。

### 3.2 四种错误码，别混

| 码 | 名字 | 什么时候用 | 你的例子 |
|---|---|---|---|
| -32700 | Parse error | 传输层 JSON 都解析不了，id 为 null | `{这不是 JSON` |
| -32600 | Invalid request | 不是合法 JSON-RPC 2.0 消息（如空批量数组） | 空数组 |
| -32601 | Method not found | **方法名**不存在 | `does/not/exist` |
| -32602 | Invalid params | 方法存在，**参数**不对 | `tools/call` 里 name 不存在、uri 不存在 |
| -32603 | Internal error | 服务端自己崩了 | 未捕获异常兜底 |

一句话记：**方法错 -32601，参数错 -32602**。
`tools/call` 传了个不存在的工具名，是「参数里的 name 错了」，不是方法错。

### 3.3 工具执行失败 ≠ 协议错误

```
工具名不存在      → JSON-RPC error -32602     （调用方写错了，模型该改）
工具跑了但查无数据 → result + isError: true    （正常业务结果，模型该读着调整）
```

代码：`ToolError` 映射成 `is_error=True`，`RpcError` 才是 JSON-RPC error。
`_ops()` 里 `if not result.ok: return _text_result(..., is_error=True)`。

**为什么**：协议错误通常意味着客户端实现有 bug，很多客户端会直接报错中断；
而「今天这家店没订单」是模型应该看到的正常信息，不该让它以为自己的调用方式错了。

### 3.4 返回必须同时有两种形态

```json
{
  "content": [{"type": "text", "text": "[1] 《客诉处理手册 …》…"}],
  "structuredContent": {"query": "...", "hits": [...]},
  "isError": false
}
```

- `content`：给模型/人读的文本，可以是 text / image / audio / resource link。
- `structuredContent`：给客户端程序读的结构，按工具的 `outputSchema` 校验（2025-06-18 新增）。

**为什么**：同一份结果，模型要读文字，前端要画表格。只给文本，客户端只能再解析一遍自然语言。

### 3.5 tools 和 resources 不是一回事

| | tools | resources |
|---|---|---|
| 谁决定用不用 | **模型**（看 description 自己判断） | **客户端/应用**（可以主动挂进上下文） |
| 标识 | name | URI |
| 类比 | 函数调用 | 文件读取 |

代码：`TOOLS` 是 8 个工具；`StoremindTools.resource_list()` 把 8 篇语料按
`storemind://corpus/<doc_id>` 暴露成资源。同一份内容两种入口，对应两种使用方式，不是重复。

### 3.6 stdio 的 stdout 只能有协议消息

**分帧方式就是「一行一条 JSON」**。所以 `print()` 进 stdout 等于往协议流里插了一行垃圾，
客户端解析失败，而报错看起来像「协议实现错了」。

代码：`serve_stdio()` 只往 stdout 写 `json.dumps(payload) + b"\n"`，
所有日志 `print(..., file=sys.stderr)`。`resource_read` 里连 `traceback` 都写 stderr。

### 3.7 HTTP 传输的三个细节

```
POST /mcp   → 请求：200 + JSON；通知：202 无 body
GET  /health→ 探活（非协议方法，方便运维）
响应头 MCP-Protocol-Version: 2025-06-18、Mcp-Session-Id
```

- 通知回 **202**（已接受、无内容），不能回 200 + `{}`——会多出一条响应。
- `Mcp-Session-Id` 让同一客户端的多次 HTTP 请求落到同一个会话。
- 2025-06-18 起，客户端**后续请求**必须带 `MCP-Protocol-Version` 头（你只发了响应头，没校验请求头，见第 8 节）。

---

## 4. 代码地图（被问到时能指到行）

| 协议概念 | 代码位置 | 说明 |
|---|---|---|
| 协议版本、服务名 | `app/mcp/server.py` 顶部常量 | `PROTOCOL_VERSION = "2025-06-18"` |
| 版本协商 | `McpServer._dispatch()` 的 `initialize` 分支 | 版本一致才回客户端版本，否则回自己的 |
| JSON-RPC 外壳 | `handle_message()` / `_handle_one()` | 校验 `jsonrpc`、区分通知、兜底异常 |
| 方法路由 | `_dispatch()` | 一张 if 表，未知方法抛 -32601 |
| 工具清单 | `TOOLS` + `TOOL_METHODS` | name/description/inputSchema 与实现一一对应 |
| 工具返回 | `_text_result()` | content + structuredContent + isError |
| 业务失败 | `ToolError` → `is_error=True` | 和 RpcError 分开 |
| 资源 | `resource_list()` / `resource_read()` | URI 前缀校验 + 路径穿越防护 |
| stdio 传输 | `serve_stdio()` | 逐行读、逐行写、解析失败 -32700 |
| HTTP 传输 | `serve_http()` | POST /mcp、202、两个响应头 |
| 协议验证 | `scripts/mcp_smoke_test.py` | 12 项，含错误码和通知无响应 |

---

## 5. 十二个追问

**Q1. MCP 和 function calling 有什么区别？**
function calling 是模型厂商的接口格式（怎么把工具描述塞进 prompt、怎么解析调用）；
MCP 是应用间的协议（工具怎么被发现、被调用、失败怎么表达、权限怎么管）。
前者解决「模型怎么选工具」，后者解决「工具从哪来」。MCP 的 server 内部完全可以用 function calling。

**Q2. 为什么要 initialize 握手，直接调工具不行吗？**
要协商两件事：协议版本（不同版本字段不同，比如 2025-06-18 新增 structuredContent、
移除了批量）和能力（服务端有没有 resources、客户端支不支持 sampling）。
不握手就调，等于假设双方字段都对得上，跨版本必炸。

**Q3. 通知和请求的区别？**
有没有 `id`。有 id 要回响应，没 id 是通知，不能回。`notifications/initialized` 就是典型。

**Q4. -32601 和 -32602 怎么分？**
看错在「方法」还是「参数」。`tools/call` 传不存在的 name 是参数错（-32602）；
请求 `foo/bar` 是方法错（-32601）。

**Q5. 工具执行失败为什么用 isError？**
因为这是业务结果，不是协议错误。协议错误会让客户端以为实现有问题；
`isError: true` 让模型读到「没查到数据」再决定怎么说，而不是直接抛异常。

**Q6. tools 和 resources 怎么选？**
需要模型判断「什么时候调」用 tools；需要客户端提前挂上下文、或按 URI 精确读取用 resources。
你的 `ask_storemind` 属于 tools，因为要不要走完整 Agent 由模型决定。

**Q7. content 和 structuredContent 为什么都要？**
一个给人/模型读，一个给程序读。只有文本，客户端得再解析自然语言；
只有结构，模型读起来费劲。

**Q8. stdio 传输为什么日志不能走 stdout？**
stdout 是协议通道，一行一条 JSON。混一行日志，客户端解析失败，
而且错误现象指向「协议实现错了」，很难想到是日志。

**Q9. HTTP 传输里为什么通知回 202？**
202 表示已接受、无内容。回 200 + `{}` 会被当成一条响应，客户端按 id 配对就乱了。

**Q10. 自己写协议和用 SDK，生产该选哪个？**
生产用 SDK：维护成本低、边界情况覆盖全。
自己写的价值是「知道协议长什么样」——客户端兼容问题、错误码语义、传输分帧，
排查时不是黑箱。这个项目的自写实现就是为了这个。

**Q11. 2025-06-18 这版改了什么？**
移除 JSON-RPC 批量、新增 structuredContent、新增 elicitation（服务端可向用户要补充信息）、
工具结果支持 resource link、HTTP 后续请求必须带 `MCP-Protocol-Version` 头、
服务端按 OAuth Resource Server 分类。**你实现了 structuredContent，批量是兼容保留，不是这版能力**——这点要说准。

**Q12. 你的实现缺什么？**
elicitation、prompts、resources/subscribe、请求头版本校验、OAuth 鉴权、SSE 流式。
主动说出来比被问出来好，还能接一句「这些在 SDK 里是现成的，所以生产我会直接用 SDK」。

---

## 6. 动手：把代码改坏，看它怎么坏

每一条都是 5 分钟，改完记得跑冒烟恢复。

1. **让通知回响应**
   `_handle_one()` 里把 `return None if is_notification else {...}` 改成永远返回。
   → 跑 `scripts/mcp_smoke_test.py`，看「通知无响应 + ping」怎么 FAIL。这就是 3.1 的现场。

2. **把未知工具改成 -32601**
   跑冒烟，「未知工具返回 -32602」那条会 FAIL。体会「方法错 vs 参数错」的边界。

3. **加一个工具**
   比如 `docs_count`：在 `TOOLS` 加一条（name/description/inputSchema），
   在 `TOOL_METHODS` 加实现，在 `StoremindTools` 加方法。跑通 `tools/list` + `tools/call`。
   → 做完你会真正理解「schema 是给模型看的，实现是给服务端跑的」。

4. **给参数加约束**
   给 `top_k` 加 `minimum/maximum`，传 100 试试——注意你的服务端不按 schema 校验，
   是在代码里 `min(top_k, 20)` 夹的。这里能讲「schema 是给模型的建议，服务端仍要自己校验」。

5. **HTTP 传输**
   ```powershell
   python scripts/mcp_server.py --http 8765
   curl.exe -s http://127.0.0.1:8765/health
   curl.exe -s -X POST http://127.0.0.1:8765/mcp -H "Content-Type: application/json" `
     -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}"
   ```
   再发一条通知（无 id），看返回码是不是 202。

6. **自己发一帧**
   ```powershell
   python scripts/mcp_play.py --step     # 每步回车，边看边想
   ```

---

## 7. 一句话总结

MCP = **JSON-RPC 2.0 + 一份能力清单 + 一套失败语义**。
真正难的从来不是方法名，而是：通知不回响应、错误码分层、结果双形态、传输通道不能混日志。
这四件事你代码里都做了，剩下的是能用话讲出来。
