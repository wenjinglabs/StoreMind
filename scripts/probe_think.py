"""ollama 生成参数对比测试。

目的：找到「不泄漏思考过程 + 速度可接受」的参数组合。
qwen3 是混合推理模型，默认会输出思维链，直接返回给用户是不能接受的。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:11434"
SYS = (
    "你是连锁酒吧的门店运营助手。规则："
    "1.只根据参考资料回答，不要编造。"
    "2.涉及金额期限比例必须原样引用数字。"
    "3.回答末尾用[1][2]标注引用来源编号。"
    "4.语气简洁，像老员工带新人，不要客套话，不要解释推理过程，直接给结论。"
)
CTX = ("[1] 《活动核销规则 · 叠加规则》满减券与折扣券不能叠加。"
       "同一订单只能选其一，系统会自动选择对客人更有利的一种，但需要服务员在结账前口头确认。")
USER = f"参考资料：{CTX}\n\n问题：满减和折扣券能一起用吗"


def call(body: dict) -> dict:
    t0 = time.perf_counter()
    with httpx.Client(timeout=180.0) as c:
        r = c.post(f"{BASE}/api/chat", json=body)
    dt = int((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"error": f"{r.status_code} {r.text[:200]}", "ms": dt}
    d = r.json()
    msg = d.get("message") or {}
    return {
        "ms": dt,
        "content": msg.get("content") or "",
        "has_thinking": "thinking" in msg,
        "thinking": (msg.get("thinking") or "")[:200],
        "eval_count": d.get("eval_count"),
        "done": d.get("done_reason"),
        "load_ms": int((d.get("load_duration") or 0) / 1_000_000),
    }


def mk(model: str, think, num_predict: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    if think is not None:
        body["think"] = think
    return body


def main() -> None:
    cases = [
        ("A. think=false, np=300", mk("qwen3:4b", False, 300)),
        ("B. think=true,  np=600", mk("qwen3:4b", True, 600)),
        ("C. 无 think 字段, np=300", mk("qwen3:4b", None, 300)),
        ("D. /no_think 前缀, np=300", {
            "model": "qwen3:4b",
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": "/no_think\n" + USER},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }),
    ]
    out: list[str] = []
    for name, body in cases:
        r = call(body)
        out.append("=" * 70)
        out.append(f"### {name}")
        if "error" in r:
            out.append(f"  ERROR: {r['error']}")
            continue
        out.append(f"  ms={r['ms']} load_ms={r['load_ms']} eval_count={r['eval_count']} done={r['done']} has_thinking={r['has_thinking']}")
        out.append(f"  content 长度: {len(r['content'])}")
        out.append("  --- content ---")
        out.append("  " + r["content"].replace("\n", "\n  "))
        if r["has_thinking"]:
            out.append("  --- thinking(截断) ---")
            out.append("  " + r["thinking"].replace("\n", "\n  "))
    text = "\n".join(out)
    Path("data/think_compare.txt").write_text(text, encoding="utf-8")
    print("written data/think_compare.txt")


if __name__ == "__main__":
    main()
