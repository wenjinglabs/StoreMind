"""工作流测试：提案 → 确认/拒绝 → 落库 → 审计。

用独立测试库 storemind_test（PostgreSQL），不碰演示库。
需要 compose 里的 postgres 正在运行。

    python scripts/workflow_test.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.workflow.db import WorkflowDB  # noqa: E402
from app.workflow.service import (  # noqa: E402
    PermissionDenied,
    ValidationFailed,
    WorkflowService,
)

TEST_DB = "storemind_test"


def _test_dsn() -> str:
    info = conninfo_to_dict(settings.pg_dsn)
    admin = dict(info, dbname="postgres")
    with psycopg.connect(make_conninfo(**admin), autocommit=True) as c:
        exists = c.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (TEST_DB,)
        ).fetchone()
        if not exists:
            c.execute(f'CREATE DATABASE "{TEST_DB}"')
    return make_conninfo(**dict(info, dbname=TEST_DB))


def main() -> int:
    db = WorkflowDB(settings, dsn=_test_dsn())
    db.reset()
    svc = WorkflowService(db)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            checks.append((name, True, str(detail or "")))
            print(f"[PASS] {name} {detail or ''}")
        except Exception as e:
            checks.append((name, False, str(e)))
            print(f"[FAIL] {name}: {e}")

    ticket = svc.propose(
        "ticket", "广州天河店",
        {"category": "酒水品质", "summary": "客人反映酒有异味，要求处理"},
        role="clerk", actor="clerk-01",
    )
    restock = svc.propose(
        "restock", "广州天河店",
        {"item_name": "莫吉托", "qty": 20, "unit": "份"},
        role="clerk", actor="clerk-01",
        idempotency_key="restock-001",
    )

    def t_propose():
        assert ticket["status"] == "pending", ticket
        assert ticket["requires_role"] == "clerk", ticket
        assert restock["requires_role"] == "manager", restock
        assert restock["risk"] == "medium", restock
        return f"ticket={ticket['action_id']} restock={restock['action_id']}"

    check("提案：状态 / 所需角色 / 风险", t_propose)

    def t_confirm_ticket():
        out = svc.confirm(ticket["action_id"], role="clerk", actor="clerk-01")
        assert out["status"] == "confirmed", out
        tid = out["result"]["ticket_id"]
        row = db.fetchone("SELECT * FROM tickets WHERE ticket_id=%s", (tid,))
        assert row and row["summary"].startswith("客诉工单"), row
        events = [e["event"] for e in svc.audit(ticket["action_id"])]
        assert events == ["proposed", "confirmed"], events
        return f"ticket_id={tid} audit={events}"

    check("确认客诉工单 + 落库 + 审计", t_confirm_ticket)

    def t_confirm_idempotent():
        before = db.fetchone("SELECT COUNT(*) AS n FROM tickets")["n"]
        out = svc.confirm(ticket["action_id"], role="clerk")
        after = db.fetchone("SELECT COUNT(*) AS n FROM tickets")["n"]
        assert out.get("already") is True, out
        assert before == after, (before, after)
        return "重复确认未重复建单"

    check("确认幂等", t_confirm_idempotent)

    def t_permission_denied():
        try:
            svc.confirm(restock["action_id"], role="clerk", actor="clerk-01")
        except PermissionDenied:
            pass
        else:
            raise AssertionError("店员确认补货应被拒绝")
        row = db.fetchone("SELECT status FROM workflow_actions WHERE action_id=%s", (restock["action_id"],))
        assert row["status"] == "pending", row
        events = [e["event"] for e in svc.audit(restock["action_id"])]
        assert "permission_denied" in events, events
        return "店员越权被拒，状态仍 pending，有审计"

    check("越权拒绝", t_permission_denied)

    def t_manager_confirm():
        out = svc.confirm(restock["action_id"], role="manager", actor="manager-01")
        assert out["status"] == "confirmed", out
        rid = out["result"]["request_id"]
        row = db.fetchone("SELECT * FROM restock_requests WHERE request_id=%s", (rid,))
        assert row and row["qty"] == 20, row
        return f"request_id={rid} qty={row['qty']}"

    check("店长确认补货 + 落库", t_manager_confirm)

    def t_idempotency_key():
        again = svc.propose(
            "restock", "广州天河店",
            {"item_name": "莫吉托", "qty": 20, "unit": "份"},
            role="clerk", actor="clerk-01",
            idempotency_key="restock-001",
        )
        assert again["action_id"] == restock["action_id"], (again, restock)
        assert again.get("deduplicated") is True, again
        return "同 key 返回同一提案"

    check("提案幂等键", t_idempotency_key)

    def t_reject():
        p = svc.propose(
            "ticket", "深圳福田店",
            {"category": "服务态度", "summary": "客人反馈服务慢"},
            role="clerk",
        )
        out = svc.reject(p["action_id"], role="clerk", actor="clerk-02", reason="信息不足")
        assert out["status"] == "rejected", out
        events = [e["event"] for e in svc.audit(p["action_id"])]
        assert events == ["proposed", "rejected"], events
        return f"events={events}"

    check("拒绝提案 + 审计", t_reject)

    def t_validation():
        try:
            svc.propose("restock", "广州天河店", {"item_name": "莫吉托", "qty": 0})
        except ValidationFailed:
            return "qty=0 被拒"
        raise AssertionError("qty=0 应被拒绝")

    check("参数校验", t_validation)

    def t_pending():
        pending = svc.list_pending()
        assert all(a["status"] == "pending" for a in pending), pending
        return f"{len(pending)} 条待办"

    check("待办列表", t_pending)

    passed = sum(1 for _, ok, _ in checks if ok)
    lines = [f"工作流测试 · {datetime.now().isoformat(timespec='seconds')}", ""]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name} · {detail}")
    lines += ["", f"结果 {passed}/{len(checks)}"]
    report = "\n".join(lines)

    out = ROOT / "data" / "workflow_report.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print()
    print(report)
    print(f"\n报告已写入 {out}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
