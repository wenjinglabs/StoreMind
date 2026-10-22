"""生成门店经营数据（结构化事实库）。

这是「规则在文档里、事实在数据库里」的那一半。
数据是构造的，但分布刻意贴近真实：周末订单量高、高单价品项拉高客单价、
活动核销集中在上半月。

运行时只写 PostgreSQL；SQLite 已移除。
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

STORES = [
    ("S001", "广州天河店", "广州"),
    ("S002", "广州珠江新城店", "广州"),
    ("S003", "深圳福田店", "深圳"),
    ("S004", "佛山祖庙店", "佛山"),
]

ITEMS = [
    ("尊尼获加黑牌", 78, 0.22),
    ("芝华士12年", 88, 0.16),
    ("格兰菲迪12年", 128, 0.10),
    ("莫吉托", 58, 0.14),
    ("长岛冰茶", 68, 0.10),
    ("威士忌酸", 62, 0.08),
    ("精酿啤酒", 45, 0.12),
    ("果盘", 38, 0.05),
    ("小食拼盘", 68, 0.03),
]

ACTIVITIES = ["满300减50", "会员日全场8折", "第二份半价"]

SCHEMA = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS redemptions;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS stores;

CREATE TABLE stores (
    store_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    city     TEXT NOT NULL
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL,
    biz_date TEXT NOT NULL,
    amount   DOUBLE PRECISION NOT NULL,
    status   TEXT NOT NULL
);
CREATE TABLE order_items (
    id        BIGSERIAL PRIMARY KEY,
    order_id  TEXT NOT NULL,
    item_name TEXT NOT NULL,
    qty       INTEGER NOT NULL,
    price     DOUBLE PRECISION NOT NULL
);
CREATE TABLE redemptions (
    id            BIGSERIAL PRIMARY KEY,
    store_id      TEXT NOT NULL,
    activity_name TEXT NOT NULL,
    redeem_date   TEXT NOT NULL,
    amount        DOUBLE PRECISION NOT NULL
);
CREATE TABLE inventory (
    store_id  TEXT NOT NULL,
    item_name TEXT NOT NULL,
    qty       INTEGER NOT NULL,
    unit      TEXT NOT NULL,
    PRIMARY KEY (store_id, item_name)
);
CREATE INDEX idx_orders_date ON orders(biz_date);
CREATE INDEX idx_orders_store ON orders(store_id);
"""


class SeedConn:
    """把生成代码里的 ? 占位符转成 psycopg 的 %s。"""

    def __init__(self, dsn: str) -> None:
        import psycopg

        self.conn = psycopg.connect(dsn)

    def executescript(self, script: str) -> None:
        with self.conn.cursor() as cur:
            for stmt in script.split(";"):
                if stmt.strip():
                    cur.execute(stmt)

    def execute(self, sql: str, params=()):
        return self.conn.execute(sql.replace("?", "%s"), params)

    def executemany(self, sql: str, rows) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(sql.replace("?", "%s"), rows)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def main(days: int = 60, seed: int = 42, dsn: str = "") -> None:
    rnd = random.Random(seed)
    conn = SeedConn(dsn or settings.pg_dsn)

    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO stores VALUES(?,?,?)", STORES)

    today = date.today()
    order_seq = 0
    total_orders = 0
    total_items = 0

    for d in range(days):
        biz = today - timedelta(days=days - 1 - d)
        is_weekend = biz.weekday() >= 4  # 周五到周日

        for store_id, name, _city in STORES:
            base = 32 if is_weekend else 16
            if store_id in ("S001", "S003"):  # 核心店更多单
                base = int(base * 1.5)
            n_orders = max(6, int(rnd.gauss(base, base * 0.25)))

            for _ in range(n_orders):
                order_seq += 1
                order_id = f"O{biz.strftime('%y%m%d')}{order_seq:05d}"
                # 每单 1~4 个品项
                picks = rnd.choices(ITEMS, weights=[w for *_, w in ITEMS], k=rnd.randint(1, 4))
                items = {}
                for item_name, price, _w in picks:
                    items[item_name] = items.get(item_name, 0) + 1
                amount = 0.0
                for item_name, price, _w in ITEMS:
                    if item_name in items:
                        qty = items[item_name]
                        # 少量客人会加杯
                        if rnd.random() < 0.15:
                            qty += 1
                        amount += qty * price
                        conn.execute(
                            "INSERT INTO order_items(order_id,item_name,qty,price) VALUES(?,?,?,?)",
                            (order_id, item_name, qty, price),
                        )
                        total_items += 1
                status = "paid" if rnd.random() > 0.03 else "void"
                conn.execute(
                    "INSERT INTO orders VALUES(?,?,?,?,?)",
                    (order_id, store_id, biz.isoformat(), round(amount, 2), status),
                )
                total_orders += 1

                # 活动核销：约 28% 的已支付订单
                if status == "paid" and rnd.random() < 0.28:
                    act = rnd.choices(ACTIVITIES, weights=[0.6, 0.25, 0.15])[0]
                    conn.execute(
                        "INSERT INTO redemptions(store_id,activity_name,redeem_date,amount) VALUES(?,?,?,?)",
                        (store_id, act, biz.isoformat(), round(rnd.uniform(30, 60), 2)),
                    )

    # 库存：刻意让部分品项低于阈值，方便演示缺货告警
    inv_rows = []
    for store_id, _name, _city in STORES:
        for item_name, _price, _w in ITEMS:
            qty = rnd.choice([6, 8, 9, 12, 15, 20, 24, 30, 40, 60, 80])
            unit = "瓶" if "牌" in item_name or "年" in item_name or "啤酒" in item_name else "份"
            inv_rows.append((store_id, item_name, qty, unit))
    conn.executemany("INSERT INTO inventory VALUES(?,?,?,?)", inv_rows)

    conn.commit()
    conn.close()

    print("ops db 生成完成: postgres")
    print(f"  门店 {len(STORES)} 家，订单 {total_orders} 单，明细 {total_items} 行，覆盖 {days} 天")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(days=args.days, seed=args.seed, dsn=args.dsn)
