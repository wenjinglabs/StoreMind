"""FastAPI 入口。

分层：api（接口） → agent（编排） → rag / tools（能力）。
接口层不写业务逻辑，只做参数校验和异常映射。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent.graph import GraphAgent
from .agent.llm import build_llm
from .agent.pipeline import Pipeline
from .config import settings
from .rag.embedder import build_embedder
from .rag.store import build_store
from .tools.ops_tools import build_ops_tools
from .workflow.db import WorkflowDB
from .workflow.service import WorkflowError, WorkflowService

app = FastAPI(
    title="StoreMind · 门店运营知识库 Agent",
    description="面向连锁酒吧/餐饮门店的 SOP 知识库问答，支持规则检索与经营数据查询的路由分发。"
                "默认走 LangGraph 编排，带 checkpointer 做多轮会话。",
    version="0.2.0",
)

settings.ensure_dirs()
_store = build_store(settings)
_embedder = build_embedder(
    settings.embed_backend, ollama_base=settings.ollama_base, model=settings.embed_model
)
_llm = build_llm(
    settings.llm_backend,
    ollama_base=settings.ollama_base,
    # openai 兼容端点的模型名跟本地 ollama 不是一套，必须分开取
    model=settings.openai_model if settings.llm_backend in ("openai", "deepseek")
    else settings.llm_model,
    base=settings.openai_base,
    key=settings.openai_key,
    timeout=settings.llm_timeout,
)
_tools = build_ops_tools(settings)
_workflow = WorkflowService(WorkflowDB(settings))
_pipeline = Pipeline(_store, _embedder, _llm, _tools, settings)
_agent = GraphAgent(_store, _embedder, _llm, _tools, settings)


# ---------- schemas ----------

class ChatIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    thread_id: str = Field(default="default", max_length=64,
                           description="会话 ID。同一个 ID 的多次调用共享上下文。")
    engine: str = Field(default="langgraph", pattern="^(langgraph|pipeline)$",
                        description="编排实现。pipeline 是手写状态机，用于 A/B 对比。")


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestIn(BaseModel):
    corpus_dir: str = Field(default="corpus")


class ActionProposalIn(BaseModel):
    kind: Literal["ticket", "restock"]
    store_name: str = Field(..., min_length=1, max_length=50)
    payload: dict = Field(default_factory=dict)
    role: str = Field(default="clerk", pattern="^(clerk|manager)$")
    actor: str = Field(default="web", max_length=50)
    idempotency_key: str | None = Field(default=None, max_length=100)


class DecisionIn(BaseModel):
    role: str = Field(default="clerk", pattern="^(clerk|manager)$")
    actor: str = Field(default="web", max_length=50)
    reason: str = Field(default="", max_length=200)


# ---------- routes ----------

@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "embed_backend": _embedder.name,
        "llm_backend": _llm.name,
        "engine": "langgraph",
        "db_backend": settings.db_backend,
        "ops_backend": settings.ops_backend,
        "db": "postgres" if settings.db_backend == "postgres" else str(settings.db_path),
        "workflow_pending": _workflow.pending_count(),
        "stats": _store.stats(),
    }


@app.get("/docs/list")
def list_docs() -> dict:
    return {"docs": _store.list_docs()}


@app.get("/docs/{doc_id}/chunks")
def doc_chunks(doc_id: str) -> dict:
    chunks = _store.get_doc_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    return {"doc_id": doc_id, "chunks": chunks}


@app.post("/search")
def search(body: SearchIn) -> dict:
    from .rag.retriever import hybrid_search

    hits = hybrid_search(_store, _embedder, body.query, top_k=body.top_k)
    return {
        "query": body.query,
        "hits": [
            {
                "doc_title": h.doc_title,
                "section_path": h.section_path,
                "score": round(h.score, 5),
                "text": h.text,
            }
            for h in hits
        ],
    }


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    state = (
        _agent.run(body.query, thread_id=body.thread_id)
        if body.engine == "langgraph"
        else _pipeline.run(body.query)
    )
    return {
        "query": state.query,
        "thread_id": body.thread_id if body.engine == "langgraph" else None,
        "engine": body.engine,
        "search_query": state.text_for_retrieval(),
        "intent": state.intent.value,
        "route_reason": state.route_reason,
        "confidence": round(state.confidence, 2),
        "answer": state.answer,
        "degraded": state.degraded,
        # Agent 层：自检结论和重查次数。
        # 前端据此显示「自检未通过 · 改写重查 N 次」——
        # 这是 Agent 和流水线的直观区别。
        "judge_ok": state.judge_ok,
        "judge_reason": state.judge_reason,
        "retries": sum(1 for s in state.steps if s.name == "rewrite"),
        "citations": state.citations,
        "data": [
            {"tool": t.tool, "ok": t.ok, "note": t.note, "data": t.data}
            for t in state.tool_results
        ],
        "trace": state.trace(),
    }


@app.get("/session/{thread_id}/history")
def session_history(thread_id: str) -> dict:
    return {"thread_id": thread_id, "turns": _agent.history(thread_id)}


# ---------- 可执行工作流 ----------
# 写操作不交给对话模型直接执行：先提案，人工确认后才落库，并写审计。

def _wf_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except WorkflowError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.post("/actions/propose")
def propose_action(body: ActionProposalIn) -> dict:
    return _wf_call(
        _workflow.propose, body.kind, body.store_name, body.payload,
        body.role, body.actor, body.idempotency_key,
    )


@app.get("/actions/pending")
def pending_actions(store_name: str = "") -> dict:
    return {"actions": _workflow.list_pending(store_name)}


@app.get("/actions/{action_id}")
def get_action(action_id: str) -> dict:
    return _wf_call(_workflow.get, action_id)


@app.get("/actions/{action_id}/audit")
def action_audit(action_id: str) -> dict:
    return {"action_id": action_id, "events": _wf_call(_workflow.audit, action_id)}


@app.post("/actions/{action_id}/confirm")
def confirm_action(action_id: str, body: DecisionIn) -> dict:
    return _wf_call(_workflow.confirm, action_id, body.role, body.actor)


@app.post("/actions/{action_id}/reject")
def reject_action(action_id: str, body: DecisionIn) -> dict:
    return _wf_call(_workflow.reject, action_id, body.role, body.actor, body.reason)


@app.delete("/session/{thread_id}")
def session_reset(thread_id: str) -> dict:
    _agent.reset(thread_id)
    return {"thread_id": thread_id, "reset": True}


@app.post("/ingest")
def ingest(body: IngestIn) -> dict:
    """重建索引。幂等：内容未变的文档会跳过。"""
    from .ingest import ingest_dir

    try:
        return ingest_dir(_store, _embedder, Path(body.corpus_dir))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- 前端页面 ----------
# 放在所有 API 路由之后注册，避免遮蔽 /docs 等接口。
# 用一个 FileResponse 而不是 mount StaticFiles：页面只有单个文件，
# 且 mount 到 "/" 会吞掉所有未匹配的 GET 请求，排查起来很烦。
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端页面缺失: app/static/index.html")
    return FileResponse(page, media_type="text/html; charset=utf-8")
