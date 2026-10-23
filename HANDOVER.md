# 交接说明 · StoreMind

> 这份文档写给**接手这个项目的 AI 或人**。
> 目标：读完这一份就能继续干活，不需要翻聊天记录。
> 文档里的「为什么」比「是什么」重要——照着代码改很容易，
> 但不知道当初为什么这么设计，就会把它改回去。

**最后更新：2026-10-22**

---

## 1. 这是什么

一个**个人作品**，不是产品。

- **技术方向**：AI 应用开发（Python）
- **选题背景**：多门店 SaaS 的运营知识通常散在群聊、SOP 文档和活动规则里。
  新人上手成本高，同一个问题不同店长答得不一样。
  所以项目主题定为**连锁酒吧/餐饮门店的运营知识库 Agent**。

**项目主题一句话**：给门店店员和店长用的 SOP 问答系统。
问「客人说酒有异味怎么办」（查文档）、问「上周天河店卖了多少」（查数据库）都能答。

---

## 2. 现在能跑什么

服务当前跑在 `http://127.0.0.1:8121/`（后台任务 `Ti7gO4`）。

```bash
cd storemind
.venv\Scripts\activate

# 起 ollama（如果没起）
# 计划任务 OllamaServe 登录自启，通常不用手动起
ollama serve

# 重建索引（换嵌入后端后会自动重建，不用手动 --rebuild）
python scripts/build_index.py

# 起应用
uvicorn app.main:app --port 8121
```

**MCP server（另一条对外入口，不依赖 FastAPI）**：

```bash
python scripts/mcp_server.py --stdio        # MCP 标准传输，客户端拉起子进程
python scripts/mcp_server.py --http 8765    # HTTP 传输，便于 curl / 容器
python scripts/mcp_smoke_test.py            # 协议冒烟，14/14
python scripts/workflow_test.py             # 工作流：提案/确认/越权/幂等/审计，9/9
```

**Docker（ollama + postgres + app + MCP 一起起）**：

```bash
docker compose -f deploy/docker-compose.yml up -d --build
python scripts/verify_docker.py --expect-embed ollama --expect-llm ollama
```

模型也在 compose 里：`ollama-import` 先从宿主机 `~/.ollama/models` 导入，`ollama-pull`
补齐缺失模型，app/mcp 走 `http://ollama:11434`。宿主端口映射 `11435`，本地脚本的
`.env` 指向它。宿主机的 `OllamaServe` 计划任务已禁用，不要再起宿主 ollama（会抢 GPU）。

**打开 `http://127.0.0.1:8000/` 是可以直接对话的网页**，
`/docs` 是 FastAPI 自动生成的接口文档。

### 四个验证命令（改完代码必须跑）

```bash
python scripts/eval.py                    # 金标集，出 recall@k 和各项指标
python scripts/smoke_test.py              # 端到端冒烟，5 条用例
python scripts/compare_orchestration.py   # 手写编排 vs LangGraph 等价性
python scripts/verify_e2e.py              # 打真模型的端到端验证，11 条用例
python scripts/mcp_smoke_test.py          # MCP 协议冒烟，14 项；加 --zero-dep 跑无模型模式（仍需 PG）
```

**当前基线（改动后如果明显低于这些数，说明你搞坏了）**：

| 项 | 值 |
|---|---|
| 规则题 recall@1 / @3 / @5 | 87.5% / 95.8% / 95.8% |
| 答案关键内容命中 | 100% |
| 数据题工具选对 / 拿到数据 | 100% / 100% |
| 无答案题正确拒答 | 100% |
| 路由准确率 | 100% |
| 编排等价性 | 6/6 |
| 冒烟 | 5/5 |
| 端到端 | 11/11 无思考泄漏 |
| MCP 协议冒烟 | 14/14（含零依赖模式） |
| Docker 部署验证 | 10/10（hashing + none 与 ollama 各一次，PG 后端） |
| 工作流测试 | 9/9（提案 / 确认 / 越权 / 幂等 / 拒绝 / 校验） |

> 全栈只剩 PostgreSQL：业务数据、工作流、会话 checkpoint 都在 PG。
> recall@1 87.5% 是单次实跑；自检重查的模型判定有波动，A/B 的 A 组 79.2% / B 组 87.5% 才是对照。

---

## 3. 环境

| 依赖 | 版本 | 位置 |
|---|---|---|
| Python | 3.13.14 | `storemind/.venv`（已建好，别重建） |
| ollama | 容器 `ollama/ollama` | 容器内 11434，宿主 11435；宿主计划任务已禁用 |
| bge-m3 | 1.2GB / 1024 维 | 已导入容器卷 |
| qwen2.5:7b | 4.7GB | 已导入容器卷 |
| qwen3:4b | 2.5GB | 已导入容器卷（**不要用它生成**，见第 6 节） |
| GPU | RTX 5060 8GB | 架构 sm_120，ollama 官方支持 |
| Docker | 29.9.2 / Desktop 4.96.1 | WSL2 后端，`hypervisorlaunchtype=Auto` |
| PostgreSQL | 17 + pgvector | 容器 `pgvector/pgvector:pg17`，宿主端口 5433 |

关键包：fastapi 0.159.1 / uvicorn 0.54.0 / pydantic 2.14.1 / httpx 0.28.1 /
langgraph 1.2.16 / langgraph-checkpoint-postgres 3.2.0

**`.env` 已配置好**（不在版本库里）：嵌入走 ollama 的 bge-m3，生成走 ollama 的 qwen2.5:7b。

> ⚠️ **`.env` 里有 key，绝不能提交。** `.gitignore` 已经覆盖了。
> 如果要新建环境，照着 `.env.example` 复制一份。

---

## 4. 架构

```
店员提问
  │
  ├─ route（正则，句式模式）──┬─ retrieve_guard ─┐
  │                          └─ data_guard ────┴─ merge ── judge ─┬─ answer
  │                                    ▲                            │
  │                                    └──── rewrite ← retrieve_again
  │                                     （唯一的环）
  └─ checkpointer（PostgreSQL，thread_id 区分会话）
```

**代码分层**（`app/`，4792 行 Python 含 scripts）：

| 文件 | 职责 | 行数 |
|---|---|---|
| `app/agent/judge.py` | **证据充分性判定 + 查询改写**（Agent 层的核心） | ~250 |
| `app/agent/pipeline.py` | 手写状态机（while 循环版） | ~560 |
| `app/agent/graph.py` | LangGraph 编排（图的环版） | ~290 |
| `app/agent/router.py` | 意图路由（纯正则，强/弱信号分层） | 112 |
| `app/agent/llm.py` | 生成层（none / ollama / openai 兼容） | 130 |
| `app/rag/*` | chunker / embedder / store（自实现 BM25）/ retriever | ~740 |
| `app/rag/pg_store.py` | PostgreSQL + pgvector（HNSW），唯一存储实现 | ~340 |
| `app/tools/pg_ops.py` | 经营数据工具（PostgreSQL） | ~110 |
| `app/workflow/db.py` | 提案 / 审计 / 工单 / 补货表（PostgreSQL） | ~110 |
| `app/workflow/service.py` | 提案 → 确认/拒绝，权限、幂等、审计 | ~260 |
| `app/tools/ops_tools.py` | 共享类型（ToolResult / Period）+ build_ops_tools 工厂 | ~50 |
| `app/mcp/server.py` | 自实现 MCP server（JSON-RPC 2.0，stdio + HTTP） | ~500 |
| `app/static/index.html` | 单文件前端，零构建 | ~530 |

**关键设计：`pipeline.py` 和 `graph.py` 是同一套逻辑的两种编排，共用节点方法。**
这是项目里最有说服力的一块——见第 6 节第 4 条。

---

## 5. 🔒 绝对不要破坏的三件事

### 5.1 无模型模式必须保留（但仍然需要 PostgreSQL）

`.env` 改两行就能退化成「不需要 GPU、不需要模型」：

```
SM_EMBED_BACKEND=hashing     # 自实现字符 n-gram 哈希嵌入
SM_LLM_BACKEND=none          # 抽取式兜底
```

**但存储只有 PostgreSQL 一个后端，SQLite 已移除。** 这套配置仍然需要
`docker compose up -d postgres`（或本地 PG），不能再声称“clone 下来什么都不装就能跑”。
`scripts/` 里所有不依赖 ollama 的脚本都要能在 hashing + none + PG 下跑通。

### 5.2 `.env` 不进版本库

里面是 opencode zen 的 key。虽然那个 key 现在用不上（免费模型 403、付费款 402），
但 key 一旦泄露是要处理的。

### 5.3 语料是「构造」的，别当真实数据改

`corpus/` 8 篇是**按「连娱宝」公开功能描述反推**造的知识域。
面试时的说法是：

> 企业知识库项目不可能拿到真实数据，所以我重点做的是语料的**分布设计**：
> 非结构化规则 + 结构化事实 + 含糊提问三类混合。

**「我怎么设计数据集」比「我有什么数据集」更能证明想过。**
改语料时保持这个结构。

---

## 6. 已知的坑（省得你重踩一遍）

按重要性排序。每一条都在代码注释和 README 里留了记录。

### 6.1 小模型会照抄格式示例 ← 最坑的一个

judge 提示词里原本写「OK 时第二行输出 `-`」，结果模型在 MISS 时也照抄 `MISS\n-`，
改写词永远是空的，**整个重查机制等于空转**（实测 10 条用例，重查修正 top1 的数量是 **0**）。

**教训**：给小模型的输出格式示例必须覆盖所有情况，
而且**代码层要有兜底**（现在兜底是从语料章节标题里挑语义最接近的当检索词）。

### 6.2 换了嵌入后端不重建索引会「静默失败」

`vector_search` 里有一句 `if len(v) != len(q): continue`——
维度不符的旧向量被**静默跳过**，表现是「召回率突然掉了」而不是报错。

**已经修好**：索引 meta 表存嵌入器签名（`hashing:ngramce09b1` / `ollama:bge-m3`），
`build_index.py` 检测到变化会强制重建并打印警告。别删这个机制。

### 6.3 多轮召回不能取并集

早先版本把几轮召回合并成一个列表（取并集），好处是「重查不会让结果变差」，
但坏处更严重：**三轮召回混在一起，模型看到的是一锅粥**。
实测「客人边吃饭边做爱怎么办」这条，top1 修对了（升级与上报），
答案却变成「现有资料里没有相关规定」——**比没有 Agent 层时更差**。

**现在改成每轮独立判定，最终只用某一轮的片段。**
如果看到有人想把几轮结果合并，直接拒绝。

### 6.4 LangGraph 的四个坑

1. **join 等的是「列表里每个节点都到过」，不是「已激活的节点」。**
   `add_edge(["a","b"], "c")` 里 b 没激活的话 c 永远等不到（答案是空字符串）。
   → 用 guard 节点（不需要的那条返回空 dict）。
2. **多入边的节点会被每条边各触发一次。** both 意图下生成跑了两次。
   → 必须有唯一汇合点。
3. **trace 不能放进 state。** state 被 checkpointer 持久化，
   `operator.add` reducer 会让 steps 跨轮次累积。→ 用 `contextvars` 在外面收集。
4. **环不能回到 join 成员上。** `retrieve_guard` 和 `data_guard` 组成 join，
   循环回 `retrieve_guard` 会破坏 join 语义。→ 单开 `retrieve_again` 节点。

### 6.5 qwen3 不能用来生成

`scripts/probe_models.py` 四种配置实测：

| 配置 | 输出 | 耗时 | 结论 |
|---|---|---|---|
| qwen3:4b `think=false` | 593 字混着「规则要求」复述 | 3.8s | 不合格 |
| qwen3:4b `think=true` | content 空，推理 1287 字没跑完被截断 | 8.1s | 不合格 |
| **qwen2.5:7b** | **39 字，干净带引用** | **401ms** | **用这个** |

附带：qwen3:4b 把「储值2000送300元」答成「赠送 2.00 元」——**小模型连「原样引用数字」
这条硬要求都做不到**。这正好是「为什么选更强的模型」的例子，README 里写了。

### 6.6 PowerShell 会毁 UTF-8 中文

`(Get-Content x -Raw) -replace ... | Set-Content` 会把中文转成 GBK 乱码。
**改文件一律用编辑工具，不要用 shell 拼。**（我犯过一次，把 verify 脚本的
正则搞坏了，排查了半天。）

同理，PowerShell 的 stdout 会被吞，脚本要用 `--out` 参数自己写 UTF-8 文件。

### 6.7 ollama 的 keep_alive

默认 5 分钟不调用就卸载模型，下次请求要重载 2.5GB 权重，延迟 0.4s → 17s。
**每次请求都显式带 `keep_alive: "30m"`**，代码里已经加了，别删。

### 6.8 Windows 上 PowerShell 首次调用可能失败

报 `CreateProcessW ERROR_ACCESS_DENIED` 时**再试一次通常就通**，
不用立刻怀疑环境坏了。

### 6.9 MCP 的 stdio 通道只能出现协议消息

`python scripts/mcp_server.py --stdio` 的 stdout 是 JSON-RPC 帧，
每隔一行一条。调试时随手 `print()` 到 stdout 会让客户端直接解析失败，
而且报错看起来像「协议实现错了」，很难往「多了一行日志」上想。
**日志一律写 stderr**，`scripts/mcp_smoke_test.py` 的报告会附上服务端 stderr，
正常不应该出现 Traceback。

另一个容易写错的点：JSON-RPC 通知（没有 `id`）**不能有响应**。
冒烟测试用「发 `notifications/initialized` 后紧接着发 ping，检查收到的 id 是 ping 的」
来验证这条。

### 6.10 LangGraph 读回旧 checkpoint 会警告业务类型未注册

state 里存了 `Hit` / `ToolResult` 两个 dataclass。langgraph 新版序列化默认只信任内置类型，
从旧 checkpoint 读回来会报：

```
Deserializing unregistered type app.rag.store.Hit from checkpoint.
```

现在 `graph.py` 里显式传了 `JsonPlusSerializer(allowed_msgpack_modules=[...])`。
**别删这两行**——新版 langgraph 会从警告变成直接拒绝，表现是「会话历史读不出来」。

### 6.11 Docker 起不来：`no virtualization available`

这台机器装 Docker Desktop 时踩了两层：

1. `wsl --install --no-distribution` 装完仍报 `no virtualization available`。
   查 `Get-ComputerInfo` / `systeminfo` 发现 `HypervisorPresent=False`，而功能项其实都是 Enabled。
2. 根因是启动配置：`bcdedit /enum {current}` 里 `hypervisorlaunchtype Off`。
   管理员执行 `bcdedit /set hypervisorlaunchtype auto` 并重启，`HypervisorPresent` 变 True，
   Docker 才真正启动。

另有一个部署隔离的坑：compose 里**不要**把宿主机 `data/` 挂进容器。
本地开发用 bge-m3 索引，容器默认 hashing，会互相覆盖嵌入器签名。
现在业务数据、工作流、checkpoint 全部在 PG，容器不再挂 `data/` 卷。

### 6.12 pgvector 的 HNSW 要求固定维度

`vector` 不带维度时 `CREATE INDEX ... USING hnsw` 会直接报
`column does not have dimensions`。现在 `PgStore` 按 `SM_EMBED_DIM` 建
`vector(512)`（hashing）或 `vector(1024)`（bge-m3）。

**启动时不再因为维度不同就删表**：零依赖/不同嵌入器可能只是临时只读访问，
向量路按 `dim` 过滤（查不到就只走 BM25）。真正换嵌入后端要跑
`python scripts/build_index.py --rebuild`，它调 `PgStore.reset()` 按当前维度重建表。
**别把维度写死成 1024**——hashing 模式要 512。

### 6.13 别让 init SQL 抢运行时 schema

之前 compose 把 `pgvector_init.sql` 挂到 `/docker-entrypoint-initdb.d`，
它先把 `docs.updated_at` 建成 `TIMESTAMPTZ`、`chunks` 建成 `vector(1024)`。
`PgStore` 的 `CREATE TABLE IF NOT EXISTS` 不会修改已有列，结果是
`stats.updated_at` 变成 datetime、列类型和代码不一致。
现在挂载已移除，运行时 schema 只由 `PgStore` 创建，`pgvector_init.sql` 只是参考。

### 6.14 小数据不是 PG 的强项

单次 `hybrid_search`（含 bge-m3 嵌入）约 60ms，45 个 chunk 下 HNSW + 网络往返
有明显固定开销。PG 的收益在并发、多实例和 10 万 chunk 以上的 ANN，不在单机小数据。
面试别说成“性能提升”，说“全栈一个库、可备份、可扩展，为规模留了路径”。

### 6.15 写操作必须走提案 → 确认，别让模型直接执行

模型会编参数（门店、品项、数量），补货/退款有权限，出了事还要能查是谁批的。
所以 `app/workflow/` 里所有写操作都拆两步：`propose_*` 只写 pending 提案，
`confirm` 才写 `tickets` / `restock_requests`，并写 `workflow_audit`。

三个不能省的约束：
- **幂等**：重复 confirm 返回第一次结果，不重复建单；提案用 `idempotency_key` 防重。
- **角色**：补货需要 `manager`；越权返回 403 并记 `permission_denied`。
- **审计**：`proposed` / `confirmed` / `rejected` / `permission_denied` 全部留痕。

注意：`role` 现在是请求字段，**只用于演示**。生产必须从 SSO / 会话里取，
不能信客户端传来的角色。这一条面试官很可能会问。

---

## 7. 接下来做什么

### 7.1 优先级（按面试价值 / 成本比排）

| 优先 | 事项 | 为什么 | 预估 |
|---|---|---|---|
| **P0** | **MCP server（自己写协议实现）** | ✅ 2026-10-22 已完成：13 个工具 + 语料资源，stdio/HTTP，协议冒烟 14/14 | 半天 |
| **P0** | **Docker 化 + 部署 + 录 3 分钟屏** | ✅ 2026-10-22 Docker 部分完成：ollama + PG + app + MCP 全在 compose，部署验证 10/10；**录屏还没做** | 1 天 |
| **P1** | **可执行工作流（受控写操作）** | ✅ 2026-10-22 已完成：客诉工单/补货，提案→确认→审计，角色/幂等，测试 9/9 | 半天 |
| **P1** | **PostgreSQL + pgvector** | ✅ 2026-10-22 已完成：HNSW，全栈 PG（知识库/经营数据/工作流/checkpoint）；45 chunk 单次检索约 60ms（别宣传成性能提升） | 半天 |
| **P0** | **改简历** | **10.27 死线**，代码再好简历上只有一行字也白搭 | 半天 |
| P1 | 两篇博客（切片对照实验 / 评估集怎么建） | 加分项第 5 条，成本极低 | 1 天 |
| P1 | 语料扩到 20 篇 + 切片对照实验 | 现有 45 个 chunk 太小，混合检索的两路互补性体现不出来 | 1 天 |
| P2 | Redis 语义缓存 | JD 有这个关键词，但「为什么用 Redis」的面试答案很无聊 | 2 小时 |
| P2 | rerank（bge-reranker） | 解决 top1 排序不理想这个最大短板 | 半天 |

### 7.2 原定排期（10.23 起）

```
10.23  MCP server
10.24  Docker + 部署 + 录屏
10.25  补语料 + 切片对照实验
10.26  两篇博客
10.27  简历改写 + 投递     ← 死线，必须留出投递和约面时间
10.28  buffer
```

**判断依据**：写代码的边际收益已经低于「能访问的链接 + 简历博客」。
如果时间不够，**先砍 P2 和 P1 的博客，不要砍简历和录屏。**

### 7.3 还没做的功能（README 路线图里）

- Redis 语义缓存
- rerank 模型
- LLM 版指代消解（现在「那退卡呢」靠规则借上文，复杂追问会失效）
- 语料扩容

---

## 8. 怎么讲这个项目

**核心叙事**（40 秒）：

> 多门店 SaaS 的痛点是运营知识散在群聊、SOP 文档和活动规则里，
> 新人上手成本高，而且同一个问题不同店长答得不一样。所以我做了一个门店知识库 Agent。
> 它不是单纯文档问答，因为门店场景里规则和事实是分开的：规则在文档里，事实在订单库里。
> 所以我做了意图路由，先判断该查文档还是该查数据库。

**六个能撑 3 分钟的技术决策**（README 第 6 节有完整版）：

1. 为什么用 pgvector 而不是独立向量库
2. 为什么不让 LLM 自由生成 SQL（注入 / 幻觉 / 权限三个角度）
3. 混合检索里两路的价值取决于嵌入模型质量（有权重对照实验）
4. 为什么先手写状态机再迁 LangGraph（同一套逻辑两种编排，6/6 等价）
5. 防幻觉的四道防线 + Agent 层的自检重查（A/B 对照 +8.3 recall@1）
6. 为什么自己实现 MCP 协议而不是调 SDK（握手 / 能力协商 / 通知无响应 / 错误码语义）
7. 为什么最终全栈上 PostgreSQL + pgvector（没有 SQLite fallback）
8. 为什么写操作要“提案 → 确认 → 审计”，而不是让 Agent 直接执行

**工作流这条怎么讲**：

> 问答 Agent 只能读，真正有用的是能做事。但写操作不能让模型直接点：
> 模型会编参数，补货要店长权限，出了事要能追溯。所以我把写操作做成提案：
> 模型/界面先 propose，人工 confirm 后才落库；补货确认要 manager，越权 403；
> 重复确认幂等，不会重复建单；proposed/confirmed/rejected 全写审计。
> 演示里可以直接在网页右上角“操作提案”点确认。
> 角色目前是请求字段，生产会从 SSO 取——这是我知道的边界。

**PG 这条怎么讲**：

> 部署和本地都用 PostgreSQL + pgvector：知识库、经营数据、工作流、会话 checkpoint
> 全在同一个库，只有一种存储，没有 SQLite fallback。向量走 HNSW，BM25 留在应用层。
> 45 个 chunk 下检索约 60ms，PG 的收益在并发、多实例和 10 万 chunk 以上的 ANN，
> 不是单机小数据——这条要主动说，别包装成“性能提升”。

**MCP 这条怎么讲**（JD 硬指标，别只说「我接入了 MCP」）：

> MCP 的传输层就是 JSON-RPC 2.0，核心是 `initialize` 协商版本和能力、`tools/list` 报清单、
> `tools/call` 执行、`resources/read` 读资源。我没用 SDK，用标准库实现了这四件事，
> 并写了协议级冒烟：通知消息不能有响应、未知方法返回 -32601、参数错误返回 -32602、
> 工具跑了但没数据返回 `isError` 而不是 JSON-RPC 错误。
> 工具分两层：原子工具给客户端自由组合，`ask_storemind` 把项目里的路由和自检重查也带过去。

**最有分量的一条**（如果面试官问「跟普通 RAG demo 有什么区别」）：

> 流水线是检索到什么都直接生成，Agent 会先自检证据够不够，不够就改写检索词重查。
> 我做了 A/B 对照：关掉这层 recall@1 是 79.2%，开启是 87.5%，答案正确率零回退，延迟只多 43 毫秒。
> 成本能压这么低是因为判定分两层——词面重合度能判的直接判，只在灰色地带才调模型。

---

## 9. GitHub 推送注意

- 默认推 private（代码默认不外露，要公开必须明确说明）
- 推送前确认 `git config user.email` 已绑定到对应的 GitHub 账号，否则贡献不算数
- 首次提交已经做了（见 `git log`），推之前检查 `.env` 没被加进去

---

## 10. 别做的事

- ❌ 别删 `pipeline.py` 说「有 LangGraph 就够了」——它是对照实验的对照组，删了就没法证明等价性
- ❌ 别把无模型模式（hashing + none）删了——它仍然需要 PG，但不需要 GPU/ollama
- ❌ 别把 `data/` 下的报告文件 gitignore 掉——那些是评估证据，面试要用
  （只有 `data/_*` 开头的临时文件该忽略）
- ❌ 别为了「简洁」把 Agent 层的自检重查去掉——它有 A/B 数据支撑
- ❌ 别在没跑完四个验证命令前说「改好了」
- ❌ 别删 `scripts/mcp_smoke_test.py` 和 `data/mcp_report*.txt`——MCP 是自己写的协议，
      这两份报告是「协议实现正确」的唯一证据
- ❌ 别删 `scripts/verify_docker.py` 和 `data/docker_verify*.txt`——Docker 部署的唯一实测证据
- ❌ 别起宿主 ollama——模型已经在 compose 的 `ollama` 服务里，两个会抢 GPU/内存
- ❌ 别把 `ollama-import` / `ollama-pull` 删了——前者从宿主缓存导入，后者补缺；删了新机器首次启动没模型
- ❌ 别把 PostgreSQL 换回 SQLite——运行时只有 PG，checkpoint 也在 PG
- ❌ 别删 `app/rag/pg_store.py` / `app/tools/pg_ops.py` / `app/workflow/`——存储和工作流的唯一实现
- ❌ 别把 `pgvector_init.sql` 重新挂到 `/docker-entrypoint-initdb.d`——会和 PgStore 抢 schema，见 6.13
- ❌ 别把 `SM_EMBED_DIM` 写死成 1024——hashing 无模型模式要 512，见 6.12
- ❌ 别删工作流的确认/权限/审计/幂等——写操作直接执行是事故，见 6.15
- ❌ 别让 MCP 的 `confirm_action` 跳过角色校验；`role` 只是演示字段，生产要从 SSO 取
- ❌ 别擅自删文件、删字段、删测试数据

---

## 附：文件速查

```
storemind/
├── README.md              ← 面试主入口，含全部设计决策和实测数据
├── HANDOVER.md            ← 这份文件
├── .env.example           ← 新环境照这个配
├── requirements.txt
├── app/
│   ├── config.py          全部可插拔配置，环境变量驱动
│   ├── ingest.py          索引构建（内容 hash 增量）
│   ├── main.py            FastAPI 入口
│   ├── agent/
│   │   ├── judge.py       证据判定 + 查询改写（Agent 核心）
│   │   ├── pipeline.py    手写状态机（含自检重查循环）
│   │   ├── graph.py       LangGraph 编排（含自检重查环）
│   │   ├── router.py      意图路由
│   │   └── llm.py         生成层
│   ├── rag/               chunker / embedder / store / retriever
│   ├── tools/ops_tools.py 白名单结构化查询
│   └── static/index.html  单文件前端
├── corpus/                8 篇构造语料（sop / rules / product）
├── eval/
│   ├── golden.json        金标集 31 条
│   └── report.md          评估报告（脚本生成）
├── scripts/               见 README 的「目录」一节
└── data/                  评估报告（面试素材，提交）
```
