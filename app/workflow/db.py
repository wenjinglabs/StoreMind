"""可执行工作流的存储层。

和 RAG 的 Store 分开：这里存的是「操作提案 / 审计 / 工单 / 补货单」，
不是文档切片。两个后端（SQLite / PostgreSQL）共用同一套 SQL，
只有自增主键和占位符不同。
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ..config import Settings


SQLITE_SCHEMA = """
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
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
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

PG_SCHEMA = SQLITE_SCHEMA.replace(
    "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY"
)


class WorkflowDB:
    def __init__(self, settings: Settings, sqlite_path: Path | None = None) -> None:
        self.backend = (settings.db_backend or "sqlite").lower()
        self.dsn = settings.pg_dsn
        self.sqlite_path = Path(sqlite_path) if sqlite_path else settings.data_dir / "workflow.sqlite3"
        self._local = threading.local()
        self._init_schema()

    # ---------- 连接 ----------

    def _conn(self):
        """线程内复用连接；写操作立即 commit。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None and not getattr(conn, "closed", 0):
            return conn

        if self.backend == "postgres":
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
        else:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.sqlite_path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
        self._local.conn = conn
        return conn

    def _q(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "postgres" else sql

    def _init_schema(self) -> None:
        schema = PG_SCHEMA if self.backend == "postgres" else SQLITE_SCHEMA
        conn = self._conn()
        if self.backend == "postgres":
            with conn.cursor() as cur:
                for stmt in schema.split(";"):
                    if stmt.strip():
                        cur.execute(stmt)
        else:
            conn.executescript(schema)

    # ---------- 通用读写 ----------

    def execute(self, sql: str, params: tuple = ()) -> None:
        conn = self._conn()
        if self.backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(self._q(sql), params)
        else:
            conn.execute(self._q(sql), params)

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        conn = self._conn()
        if self.backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(self._q(sql), params)
                row = cur.fetchone()
            return dict(row) if row else None
        row = conn.execute(self._q(sql), params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._conn()
        if self.backend == "postgres":
            with conn.cursor() as cur:
                cur.execute(self._q(sql), params)
                return [dict(r) for r in cur.fetchall()]
        return [dict(r) for r in conn.execute(self._q(sql), params).fetchall()]

    def reset(self) -> None:
        """测试用：清空工作流相关表，不动 RAG 索引。"""
        for t in ("workflow_audit", "workflow_actions", "tickets", "restock_requests"):
            self.execute(f"DELETE FROM {t}")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None and not getattr(conn, "closed", 0):
            conn.close()
