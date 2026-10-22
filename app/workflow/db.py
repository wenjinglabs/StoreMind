"""可执行工作流的存储层（PostgreSQL）。

存的是「操作提案 / 审计 / 工单 / 补货单」，不是文档切片。
运行时只有一个后端；SQLite 已移除。
"""
from __future__ import annotations

import threading

import psycopg
from psycopg.rows import dict_row

from ..config import Settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_actions (
    action_id       TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    status          TEXT NOT NULL,
    store_name      TEXT NOT NULL,
    summary         TEXT NOT NULL,
    payload         TEXT NOT NULL,
    risk            TEXT NOT NULL,
    requires_role   TEXT NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    decided_by      TEXT,
    decided_at      TEXT,
    result          TEXT,
    idempotency_key TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_workflow_idem
    ON workflow_actions(idempotency_key);
CREATE TABLE IF NOT EXISTS workflow_audit (
    id         BIGSERIAL PRIMARY KEY,
    action_id  TEXT NOT NULL,
    event      TEXT NOT NULL,
    actor      TEXT NOT NULL,
    detail     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_action ON workflow_audit(action_id);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id  TEXT PRIMARY KEY,
    store_name TEXT NOT NULL,
    category   TEXT NOT NULL,
    summary    TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS restock_requests (
    request_id TEXT PRIMARY KEY,
    store_name TEXT NOT NULL,
    item_name  TEXT NOT NULL,
    qty        INTEGER NOT NULL,
    unit       TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class WorkflowDB:
    def __init__(self, settings: Settings, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self._local = threading.local()
        self._init_schema()

    # ---------- 连接 ----------

    def _conn(self) -> psycopg.Connection:
        """线程内复用连接；写操作立即 commit。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            return conn
        conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
        self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._conn().cursor() as cur:
            for stmt in SCHEMA.split(";"):
                if stmt.strip():
                    cur.execute(stmt)

    # ---------- 通用读写 ----------

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._conn().cursor() as cur:
            cur.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._conn().cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn().cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def reset(self) -> None:
        """测试用：清空工作流相关表，不动 RAG 索引。"""
        for t in ("workflow_audit", "workflow_actions", "tickets", "restock_requests"):
            self.execute(f"DELETE FROM {t}")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            conn.close()
