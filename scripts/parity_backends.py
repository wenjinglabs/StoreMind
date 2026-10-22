"""SQLite ↔ PostgreSQL 迁移对照实验。

同一个语料、同一个嵌入器，分别建一份 SQLite 索引和一份 PG(pgvector) 索引，
然后比对三件事：

1. 索引规模：docs / chunks / terms 是否一致
2. 规则题召回：24 条金标 policy 题的 top-5 章节是否一致
3. 经营数据：5 个白名单工具在相同参数下的返回是否逐行一致

这不是「感觉迁移没问题」，是可复现的对照。报告写到 data/parity_report.txt。

    python scripts/parity_backends.py
    python scripts/parity_backends.py --embed-backend hashing
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ingest import ingest_dir  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.pg_store import PgStore  # noqa: E402
from app.rag.retriever import hybrid_search  # noqa: E402
from app.rag.store import Store  # noqa: E402
from app.tools.ops_tools import OpsTools  # noqa: E402
from app.tools.pg_ops import PgOpsTools  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.json"


def _ids(hits) -> list[str]:
    return [h.section_path for h in hits]


def _norm(value):
    """把 dict/list 里的 Decimal 等转成可比较的普通类型。"""
    return json.loads(json.dumps(value, ensure_ascii=False, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-backend", default="", help="hashing | ollama，默认读 .env")
    ap.add_argument("--out", default="data/parity_report.txt")
    args = ap.parse_args()

    settings.ensure_dirs()
    backend = args.embed_backend or settings.embed_backend
    embedder = build_embedder(
        backend, ollama_base=settings.ollama_base, model=settings.embed_model
    )
    dim = 512 if embedder.name == "hashing" else 1024
    sig = embedder.signature()

    # 临时 SQLite 库：不碰本地正在用的 data/storemind.sqlite3
    sqlite_path = settings.data_dir / "_parity_sqlite.sqlite3"
    for p in (sqlite_path, Path(str(sqlite_path) + "-wal"), Path(str(sqlite_path) + "-shm")):
        if p.exists():
            p.unlink()
    sqlite_store = Store(sqlite_path)
    sqlite_store.reset()
    corpus = ROOT / "corpus"
    ingest_dir(sqlite_store, embedder, corpus)
    sqlite_store.set_meta("embedder_sig", sig)

    # PG 索引用当前 dsn，重建后再测
    pg_store = PgStore(settings.pg_dsn, dim=dim)
    pg_store.reset()
    ingest_dir(pg_store, embedder, corpus)
    pg_store.set_meta("embedder_sig", sig)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    s1, s2 = sqlite_store.stats(), pg_store.stats()
    check("索引规模", all(s1[k] == s2[k] for k in ("docs", "chunks", "terms")),
          f"sqlite={s1} · pg={s2}")

    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    policy = [c for c in cases if c["type"] == "policy"]

    top1_same = 0
    set_same = 0
    diffs: list[str] = []
    for c in policy:
        h1 = hybrid_search(sqlite_store, embedder, c["query"], top_k=5)
        h2 = hybrid_search(pg_store, embedder, c["query"], top_k=5)
        a, b = _ids(h1), _ids(h2)
        top1_same += bool(a and b and a[0] == b[0])
        set_same += set(a) == set(b)
        if a != b:
            diffs.append(f"{c['id']} {c['query']}\n  sqlite={a}\n  pg    ={b}")

    n = len(policy)
    check("规则题 top1 一致", top1_same == n, f"{top1_same}/{n}")
    check("规则题召回集合一致", set_same == n, f"{set_same}/{n}")

    # 经营数据逐工具对照
    sqlite_tools = OpsTools(settings.data_dir / "ops.sqlite3")
    pg_tools = PgOpsTools(settings.pg_dsn)
    cases_ops = [
        ("list_stores", lambda t: t.list_stores()),
        ("store_sales", lambda t: t.store_sales("广州天河店", "last_7d")),
        ("top_items", lambda t: t.top_items("广州天河店", "last_7d", 5)),
        ("redemption_stats", lambda t: t.redemption_stats("广州天河店", "last_7d")),
        ("inventory_alert", lambda t: t.inventory_alert("广州天河店", 10)),
    ]
    ops_ok = 0
    ops_diffs: list[str] = []
    for name, fn in cases_ops:
        r1, r2 = fn(sqlite_tools), fn(pg_tools)
        same = (r1.ok == r2.ok and _norm(r1.data) == _norm(r2.data))
        ops_ok += same
        if not same:
            ops_diffs.append(f"{name}\n  sqlite={r1.data}\n  pg    ={r2.data}")
    check("经营工具输出一致", ops_ok == len(cases_ops), f"{ops_ok}/{len(cases_ops)}")

    # 延迟只做参考，不做断言：PG 走网络/进程边界，天然比本地文件慢一点
    import time
    q = policy[0]["query"]
    t0 = time.perf_counter()
    for _ in range(10):
        hybrid_search(sqlite_store, embedder, q, top_k=5)
    t_sql = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        hybrid_search(pg_store, embedder, q, top_k=5)
    t_pg = (time.perf_counter() - t0) / 10 * 1000

    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [
        f"SQLite ↔ PostgreSQL 迁移对照 · {datetime.now().isoformat(timespec='seconds')}",
        f"embedder={embedder.name} dim={dim} corpus={corpus}",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name} · {detail}")
    lines.append("")
    lines.append(f"单次 hybrid_search 平均延迟（10 次，含嵌入）：sqlite={t_sql:.1f}ms · pg={t_pg:.1f}ms")
    if diffs:
        lines += ["", "召回差异：", *diffs[:5]]
    if ops_diffs:
        lines += ["", "经营数据差异：", *ops_diffs]
    lines += ["", f"结果 {passed}/{len(checks)}"]
    report = "\n".join(lines)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n报告已写入 {out}")

    # 临时 SQLite 文件保留在 data/_parity_sqlite.sqlite3：
    # SQLite 连接由各方法内部打开，Windows 上句柄释放有延迟，删文件会偶发 WinError 32。
    # 文件名以 _ 开头，已被 .gitignore 覆盖。
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
