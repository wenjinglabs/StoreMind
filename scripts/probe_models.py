"""生成模型选型对照。

问题：qwen3 是混合推理模型，think=true 时推理进 thinking 字段、content 只剩
最终答案（但推理会吃光 token 预算导致 content 为空）；think=false 时推理
直接混进 content，用户会看到一大段「首先我需要...」。
所以要测：qwen3:4b 和 qwen2.5:7b（纯 instruct）在同一 prompt 下的
「答案是否干净」和「耗时」。

判分标准写死，避免我事后挑结果：
- 干净：content 里不出现「首先」「我需要」「根据参考资料」「思考」等痕迹
- 有引用：包含 [1] 这样的引用标记
- 简洁：长度 <= 200 字
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:11434"

SYS = """你是连锁酒吧的门店运营助手，服务对象是门店店员和店长。

规则：
1. 只根据提供的「参考资料」和「经营数据」回答，不要编造。
2. 如果资料里没有答案，直接说「现有资料里没有相关规定」，并建议联系谁。
3. 涉及金额、期限、比例、数量时，必须原样引用资料中的数字。
4. 回答末尾用 [1] [2] 标注引用来源编号。
5. 语气简洁，像老员工带新人，不要客套话。

输出格式（必须严格遵守）：
- 只输出最终答案正文，一到三句话。写完就停。
- 不要复述上面的规则，不要复述问题，不要写「首先」「我需要」「根据参考资料」这类开场白。
- 不要输出思考过程、推理步骤或自我检查。
- 不要用列表、标题或加粗，只写连贯的段落。

示例（合格）：
满减券和折扣券不能叠加，同一订单只能选其一，需要服务员在结账前口头确认。[1]

示例（不合格）：
首先，用户问的是满减和折扣券能否一起使用。根据资料[1]，规则明确说明不能叠加……
"""

CTX = ("[1] 《活动核销规则 · 叠加规则》满减券与折扣券不能叠加。同一订单只能选其一，"
       "系统会自动选择对客人更有利的一种，但需要服务员在结账前口头确认。\n"
       "[2] 《会员卡与储值规则 · 储值与赠送》储值 2000 元赠送 200 元，赠送金额不可提现。")
USER = f"参考资料：\n{CTX}\n\n问题：满减和折扣券能一起用吗"

DIRTY = re.compile(r"(首先|我需要|根据参考资料|根据\[|让我|思考|分析如下|回顾|综上|检查是否|回答结构)")


def call(model: str, think, np: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": np},
    }
    if think is not None:
        body["think"] = think
    t0 = time.perf_counter()
    with httpx.Client(timeout=180.0) as c:
        r = c.post(f"{BASE}/api/chat", json=body)
    ms = int((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"error": f"{r.status_code} {r.text[:150]}", "ms": ms}
    d = r.json()
    msg = d.get("message") or {}
    content = (msg.get("content") or "").strip()
    return {
        "ms": ms,
        "content": content,
        "thinking_len": len(msg.get("thinking") or ""),
        "eval_count": d.get("eval_count"),
        "done": d.get("done_reason"),
        "load_ms": int((d.get("load_duration") or 0) / 1_000_000),
    }


def score(r: dict) -> dict:
    if "error" in r:
        return {"clean": False, "cited": False, "brief": False}
    c = r["content"]
    return {
        "clean": not DIRTY.search(c),
        "cited": bool(re.search(r"\[\d+\]", c)),
        "brief": len(c) <= 200,
    }


def main() -> None:
    cases = [
        ("qwen3:4b   think=false np=400", "qwen3:4b", False, 400),
        ("qwen3:4b   think=true  np=900", "qwen3:4b", True, 900),
        ("qwen2.5:7b 无think    np=400", "qwen2.5:7b", None, 400),
        ("qwen2.5:7b think=false np=400", "qwen2.5:7b", False, 400),
    ]
    out: list[str] = []
    for name, model, think, np in cases:
        r = call(model, think, np)
        out.append("=" * 72)
        out.append(f"### {name}")
        if "error" in r:
            out.append(f"  ERROR {r['error']}")
            continue
        s = score(r)
        out.append(f"  ms={r['ms']} (load {r['load_ms']})  eval_count={r['eval_count']} done={r['done']} thinking_len={r['thinking_len']}")
        out.append(f"  判定: 干净={s['clean']}  有引用={s['cited']}  简洁={s['brief']}  长度={len(r['content'])}")
        out.append("  --- content ---")
        out.append("  " + r["content"].replace("\n", "\n  "))
    text = "\n".join(out)
    Path("data/model_compare.txt").write_text(text, encoding="utf-8")
    print("written data/model_compare.txt")


if __name__ == "__main__":
    main()
