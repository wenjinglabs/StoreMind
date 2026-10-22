"""受控写操作：提案 → 人工确认/拒绝 → 落库 + 审计。

为什么不让模型直接执行写操作：
1. 模型会编参数（门店、品项、数量），写错就是真实业务事故；
2. 需要权限：补货/退款不是店员能单独定的；
3. 需要可追溯：谁提的、谁批的、改了什么，审计表必须留痕。

所以所有写操作都拆成两步：
- `propose_*` 只生成一条 pending 记录，不产生业务副作用；
- `confirm_*` 才真正写 tickets / restock_requests，并且先过角色校验。

确认是幂等的：重复确认同一条 action 返回第一次的结果，不会重复建单。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .db import WorkflowDB


class WorkflowError(Exception):
    status_code = 400


class NotFound(WorkflowError):
    status_code = 404


class PermissionDenied(WorkflowError):
    status_code = 403


class Conflict(WorkflowError):
    status_code = 409


class ValidationFailed(WorkflowError):
    status_code = 422


ROLE_RANK = {"clerk": 1, "manager": 2}
KINDS = ("ticket", "restock")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


class WorkflowService:
    def __init__(self, db: WorkflowDB) -> None:
        self.db = db

    # ---------- 提案 ----------

    def propose(
        self,
        kind: str,
        store_name: str,
        payload: dict[str, Any],
        role: str = "clerk",
        actor: str = "web",
        idempotency_key: str | None = None,
    ) -> dict:
        kind = (kind or "").strip()
        store_name = (store_name or "").strip()
        role = (role or "clerk").strip().lower()
        if kind not in KINDS:
            raise ValidationFailed(f"kind 只能是 {KINDS}")
        if not store_name:
            raise ValidationFailed("store_name 不能为空")
        if role not in ROLE_RANK:
            raise ValidationFailed(f"role 只能是 {tuple(ROLE_RANK)}")

        if idempotency_key:
            existing = self.db.fetchone(
                "SELECT * FROM workflow_actions WHERE idempotency_key = %s",
                (idempotency_key,),
            )
            if existing:
                out = self._public(existing)
                out["deduplicated"] = True
                return out

        summary, requires_role, risk = self._describe(kind, store_name, payload)
        action_id = _new_id("ACT")
        now = _now()
        self.db.execute(
            """INSERT INTO workflow_actions
               (action_id,kind,status,store_name,summary,payload,risk,requires_role,
                created_by,created_at,idempotency_key)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                action_id, kind, "pending", store_name, summary,
                json.dumps(payload, ensure_ascii=False), risk, requires_role,
                actor, now, idempotency_key,
            ),
        )
        self._audit(action_id, "proposed", actor, summary)

        return self._public(self.db.fetchone(
            "SELECT * FROM workflow_actions WHERE action_id = %s", (action_id,)
        ))

    def _describe(self, kind: str, store_name: str, payload: dict) -> tuple[str, str, str]:
        if kind == "ticket":
            category = str(payload.get("category") or "客诉").strip()
            text = str(payload.get("summary") or "").strip()
            if not text:
                raise ValidationFailed("ticket 需要 payload.summary")
            summary = f"客诉工单 · {category} · {store_name}：{text[:60]}"
            # 客诉先建单，店员即可确认；涉及赔付时由店长在业务系统里再审批。
            return summary, "clerk", "low"

        item = str(payload.get("item_name") or "").strip()
        if not item:
            raise ValidationFailed("restock 需要 payload.item_name")
        try:
            qty = int(payload.get("qty") or 0)
        except (TypeError, ValueError):
            raise ValidationFailed("payload.qty 必须是整数")
        if qty <= 0:
            raise ValidationFailed("payload.qty 必须大于 0")
        unit = str(payload.get("unit") or "份").strip()
        risk = "high" if qty >= 50 else "medium"
        summary = f"补货申请 · {store_name} · {item} × {qty}{unit}"
        return summary, "manager", risk

    # ---------- 确认 / 拒绝 ----------

    def confirm(self, action_id: str, role: str = "clerk", actor: str = "web") -> dict:
        action = self._get(action_id)
        role = (role or "clerk").strip().lower()

        if action["status"] == "confirmed":
            out = self._public(action)
            out["already"] = True
            return out
        if action["status"] == "rejected":
            raise Conflict("该提案已被拒绝，不能确认")

        if ROLE_RANK.get(role, 0) < ROLE_RANK[action["requires_role"]]:
            self._audit(action_id, "permission_denied", actor,
                        f"role={role} 需要 {action['requires_role']}")
            raise PermissionDenied(
                f"需要 {action['requires_role']} 角色才能确认，当前 role={role}"
            )

        payload = json.loads(action["payload"])
        now = _now()
        if action["kind"] == "ticket":
            result = {
                "ticket_id": _new_id("T"),
                "status": "open",
                "category": payload.get("category") or "客诉",
            }
            self.db.execute(
                "INSERT INTO tickets(ticket_id,store_name,category,summary,status,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (result["ticket_id"], action["store_name"], result["category"],
                 action["summary"], "open", now),
            )
        else:
            result = {
                "request_id": _new_id("R"),
                "status": "pending_fulfillment",
                "item_name": payload.get("item_name"),
                "qty": int(payload.get("qty")),
            }
            self.db.execute(
                "INSERT INTO restock_requests(request_id,store_name,item_name,qty,unit,status,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (result["request_id"], action["store_name"], result["item_name"],
                 result["qty"], payload.get("unit") or "份", "pending_fulfillment", now),
            )

        self.db.execute(
            "UPDATE workflow_actions SET status=%s, decided_by=%s, decided_at=%s, result=%s "
            "WHERE action_id=%s",
            ("confirmed", actor, now, json.dumps(result, ensure_ascii=False), action_id),
        )
        self._audit(action_id, "confirmed", actor, json.dumps(result, ensure_ascii=False))
        return self._public(self._get(action_id))

    def reject(self, action_id: str, role: str = "clerk", actor: str = "web",
               reason: str = "") -> dict:
        action = self._get(action_id)
        role = (role or "clerk").strip().lower()
        if action["status"] == "confirmed":
            raise Conflict("该提案已确认，不能拒绝")
        if action["status"] == "rejected":
            out = self._public(action)
            out["already"] = True
            return out
        if ROLE_RANK.get(role, 0) < ROLE_RANK[action["requires_role"]]:
            self._audit(action_id, "permission_denied", actor,
                        f"role={role} 需要 {action['requires_role']}")
            raise PermissionDenied(
                f"需要 {action['requires_role']} 角色才能拒绝，当前 role={role}"
            )

        now = _now()
        result = {"reason": reason or "未填写原因"}
        self.db.execute(
            "UPDATE workflow_actions SET status=%s, decided_by=%s, decided_at=%s, result=%s "
            "WHERE action_id=%s",
            ("rejected", actor, now, json.dumps(result, ensure_ascii=False), action_id),
        )
        self._audit(action_id, "rejected", actor, result["reason"])
        return self._public(self._get(action_id))

    # ---------- 查询 ----------

    def list_pending(self, store_name: str = "") -> list[dict]:
        if store_name:
            rows = self.db.fetchall(
                "SELECT * FROM workflow_actions WHERE status='pending' AND store_name LIKE %s "
                "ORDER BY created_at DESC",
                (f"%{store_name}%",),
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM workflow_actions WHERE status='pending' "
                "ORDER BY created_at DESC"
            )
        return [self._public(r) for r in rows]

    def pending_count(self) -> int:
        row = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM workflow_actions WHERE status='pending'"
        )
        return int(row["n"]) if row else 0

    def get(self, action_id: str) -> dict:
        return self._public(self._get(action_id))

    def audit(self, action_id: str) -> list[dict]:
        self._get(action_id)  # 404 if missing
        return self.db.fetchall(
            "SELECT event,actor,detail,created_at FROM workflow_audit "
            "WHERE action_id=%s ORDER BY id",
            (action_id,),
        )

    # ---------- 内部 ----------

    def _get(self, action_id: str) -> dict:
        row = self.db.fetchone(
            "SELECT * FROM workflow_actions WHERE action_id = %s", (action_id,)
        )
        if not row:
            raise NotFound(f"提案不存在: {action_id}")
        return row

    def _audit(self, action_id: str, event: str, actor: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO workflow_audit(action_id,event,actor,detail,created_at) "
            "VALUES(%s,%s,%s,%s,%s)",
            (action_id, event, actor, detail, _now()),
        )

    @staticmethod
    def _public(row: dict | None) -> dict:
        if not row:
            raise NotFound("提案不存在")
        out = dict(row)
        for key in ("payload", "result"):
            if isinstance(out.get(key), str):
                try:
                    out[key] = json.loads(out[key])
                except json.JSONDecodeError:
                    pass
        return out
