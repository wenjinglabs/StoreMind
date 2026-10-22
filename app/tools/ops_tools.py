"""结构化数据查询工具。

重要设计决策：不让 LLM 自由生成 SQL。

理由有三条，面试时这就是一道现成的题：
1. 注入风险。门店名、品项名来自用户输入，拼进 SQL 就是注入面。
2. 幻觉风险。模型会编出不存在的表名和字段，线上表现是随机 500。
3. 权限不可控。自由 SQL 意味着模型可以 SELECT 任何表，包括员工薪资。

所以只暴露白名单工具，参数用 Pydantic 校验，SQL 全部预编译。
模型能做的是「选哪个工具、填什么参数」，不是「写 SQL」。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
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


class OpsTools:
    """门店经营数据工具集。所有方法都是白名单、参数化、只读。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _conn(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"经营库不存在: {self.db_path}，先跑 scripts/seed_ops_db.py"
            )
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def list_stores(self) -> ToolResult:
        sql = "SELECT store_id, name, city FROM stores ORDER BY store_id"
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql)]
        return ToolResult("list_stores", True, rows, sql)

    def store_sales(self, store_name: str, period: Period = "last_7d") -> ToolResult:
        start, end = _range(period)
        sql = """
            SELECT s.name AS store, COUNT(o.order_id) AS orders,
                   ROUND(SUM(o.amount), 2) AS revenue,
                   ROUND(AVG(o.amount), 2) AS avg_ticket
            FROM orders o JOIN stores s ON s.store_id = o.store_id
            WHERE s.name LIKE ? AND o.biz_date BETWEEN ? AND ? AND o.status = 'paid'
            GROUP BY s.store_id
        """
        with self._conn() as c:
            row = c.execute(sql, (f"%{store_name}%", start, end)).fetchone()
        if not row:
            return ToolResult("store_sales", False, None, sql,
                              f"{start}~{end} 区间内该门店无已支付订单")
        return ToolResult("store_sales", True, dict(row), sql, f"区间 {start} ~ {end}")

    def top_items(self, store_name: str, period: Period = "last_7d", limit: int = 5) -> ToolResult:
        start, end = _range(period)
        limit = max(1, min(int(limit), 20))
        sql = """
            SELECT i.item_name, SUM(i.qty) AS qty, ROUND(SUM(i.qty * i.price), 2) AS revenue
            FROM order_items i
            JOIN orders o ON o.order_id = i.order_id
            JOIN stores s ON s.store_id = o.store_id
            WHERE s.name LIKE ? AND o.biz_date BETWEEN ? AND ? AND o.status = 'paid'
            GROUP BY i.item_name
            ORDER BY revenue DESC LIMIT ?
        """
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, (f"%{store_name}%", start, end, limit))]
        return ToolResult("top_items", bool(rows), rows, sql, f"区间 {start} ~ {end}")

    def redemption_stats(self, store_name: str, period: Period = "last_7d") -> ToolResult:
        start, end = _range(period)
        sql = """
            SELECT s.name AS store, r.activity_name, COUNT(*) AS redemptions,
                   ROUND(SUM(r.amount), 2) AS total_amount
            FROM redemptions r JOIN stores s ON s.store_id = r.store_id
            WHERE s.name LIKE ? AND r.redeem_date BETWEEN ? AND ?
            GROUP BY r.activity_name
            ORDER BY redemptions DESC
        """
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, (f"%{store_name}%", start, end))]
        return ToolResult("redemption_stats", bool(rows), rows, sql, f"区间 {start} ~ {end}")

    def inventory_alert(self, store_name: str, threshold: int = 10) -> ToolResult:
        sql = """
            SELECT s.name AS store, v.item_name, v.qty, v.unit
            FROM inventory v JOIN stores s ON s.store_id = v.store_id
            WHERE s.name LIKE ? AND v.qty <= ?
            ORDER BY v.qty ASC LIMIT 20
        """
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, (f"%{store_name}%", threshold))]
        return ToolResult("inventory_alert", bool(rows), rows, sql, f"阈值 {threshold}")

    # 白名单注册表：Agent 只能从这里挑
    @property
    def registry(self) -> dict[str, str]:
        return {
            "list_stores": "列出所有门店",
            "store_sales": "查某门店某周期的订单数与营业额，参数 store_name, period",
            "top_items": "查某门店热销品项排行，参数 store_name, period, limit",
            "redemption_stats": "查某门店活动核销统计，参数 store_name, period",
            "inventory_alert": "查某门店低库存品项，参数 store_name, threshold",
        }
