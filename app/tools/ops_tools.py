"""结构化数据查询的共享类型与工厂。

运行时只有 PostgreSQL 一个实现（`pg_ops.py`）。
这里保留 `Period` / `ToolResult` / `_range`，因为 PgOpsTools 和上层都依赖它们；
SQLite 实现已经移除。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

Period = Literal["today", "yesterday", "last_7d", "last_30d", "this_month"]


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: object
    sql: str
    note: str = ""


def _range(period: Period) -> tuple[str, str]:
    today = date.today()
    if period == "today":
        return today.isoformat(), today.isoformat()
    if period == "yesterday":
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if period == "last_7d":
        return (today - timedelta(days=6)).isoformat(), today.isoformat()
    if period == "last_30d":
        return (today - timedelta(days=29)).isoformat(), today.isoformat()
    first = today.replace(day=1)
    return first.isoformat(), today.isoformat()


def build_ops_tools(settings):
    """经营数据只有一个后端：PostgreSQL。"""
    from .pg_ops import PgOpsTools

    return PgOpsTools(settings.pg_dsn)
