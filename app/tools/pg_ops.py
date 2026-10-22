"""经营数据工具（PostgreSQL 版）。

和 `ops_tools.py` 同一组白名单、同一套 SQL，只换了驱动和占位符。
仍然不让模型生成 SQL：参数用 Pydantic 在 API 层校验，这里只接受
枚举值（period）和已经收敛的字符串（store_name / threshold）。
"""
from __future__ import annotations

import threading

import psycopg
from psycopg.rows import dict_row

from .ops_tools import ToolResult, _range


class PgOpsTools:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._local = threading.local()

    def _conn(self) -> psycopg.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)
            self._local.conn = conn
        return conn

    def list_stores(self) -> ToolResult:
        sql = "SELECT store_id, name, city FROM stores ORDER BY store_id"
        with self._conn().cursor() as cur:
            cur.execute(sql)
            rows = list(cur.fetchall())
        return ToolResult("list_stores", True, rows, sql)

    def store_sales(self, store_name: str, period: str = "last_7d") -> ToolResult:
        start, end = _range(period)  # type: ignore[arg-type]
        sql = """
            SELECT s.name AS store, COUNT(o.order_id) AS orders,
                   ROUND(SUM(o.amount)::numeric, 2)::float8 AS revenue,
                   ROUND(AVG(o.amount)::numeric, 2)::float8 AS avg_ticket
            FROM orders o JOIN stores s ON s.store_id = o.store_id
            WHERE s.name LIKE %s AND o.biz_date BETWEEN %s AND %s AND o.status = 'paid'
            GROUP BY s.store_id, s.name
        """
        with self._conn().cursor() as cur:
            cur.execute(sql, (f"%{store_name}%", start, end))
            row = cur.fetchone()
        if not row:
            return ToolResult("store_sales", False, None, sql,
                              f"{start}~{end} 区间内该门店无已支付订单")
        return ToolResult("store_sales", True, dict(row), sql, f"区间 {start} ~ {end}")

    def top_items(self, store_name: str, period: str = "last_7d", limit: int = 5) -> ToolResult:
        start, end = _range(period)  # type: ignore[arg-type]
        limit = max(1, min(int(limit), 20))
        sql = """
            SELECT i.item_name, SUM(i.qty)::int AS qty,
                   ROUND(SUM(i.qty * i.price)::numeric, 2)::float8 AS revenue
            FROM order_items i
            JOIN orders o ON o.order_id = i.order_id
            JOIN stores s ON s.store_id = o.store_id
            WHERE s.name LIKE %s AND o.biz_date BETWEEN %s AND %s AND o.status = 'paid'
            GROUP BY i.item_name
            ORDER BY revenue DESC LIMIT %s
        """
        with self._conn().cursor() as cur:
            cur.execute(sql, (f"%{store_name}%", start, end, limit))
            rows = list(cur.fetchall())
        return ToolResult("top_items", bool(rows), rows, sql, f"区间 {start} ~ {end}")

    def redemption_stats(self, store_name: str, period: str = "last_7d") -> ToolResult:
        start, end = _range(period)  # type: ignore[arg-type]
        sql = """
            SELECT s.name AS store, r.activity_name, COUNT(*)::int AS redemptions,
                   ROUND(SUM(r.amount)::numeric, 2)::float8 AS total_amount
            FROM redemptions r JOIN stores s ON s.store_id = r.store_id
            WHERE s.name LIKE %s AND r.redeem_date BETWEEN %s AND %s
            GROUP BY r.activity_name, s.name
            ORDER BY redemptions DESC
        """
        with self._conn().cursor() as cur:
            cur.execute(sql, (f"%{store_name}%", start, end))
            rows = list(cur.fetchall())
        return ToolResult("redemption_stats", bool(rows), rows, sql, f"区间 {start} ~ {end}")

    def inventory_alert(self, store_name: str, threshold: int = 10) -> ToolResult:
        sql = """
            SELECT s.name AS store, v.item_name, v.qty, v.unit
            FROM inventory v JOIN stores s ON s.store_id = v.store_id
            WHERE s.name LIKE %s AND v.qty <= %s
            ORDER BY v.qty ASC LIMIT 20
        """
        with self._conn().cursor() as cur:
            cur.execute(sql, (f"%{store_name}%", threshold))
            rows = list(cur.fetchall())
        return ToolResult("inventory_alert", bool(rows), rows, sql, f"阈值 {threshold}")

    @property
    def registry(self) -> dict[str, str]:
        return {
            "list_stores": "列出所有门店",
            "store_sales": "查某门店某周期的订单数与营业额，参数 store_name, period",
            "top_items": "查某门店热销品项排行，参数 store_name, period, limit",
            "redemption_stats": "查某门店活动核销统计，参数 store_name, period",
            "inventory_alert": "查某门店低库存品项，参数 store_name, threshold",
        }
