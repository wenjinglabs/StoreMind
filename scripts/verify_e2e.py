"""端到端真实验证：走 HTTP 接口，打真模型，逐条看答案质量。

不是单元测试，是「面试官会问什么」的预演：
每条问题都人工看一遍答案对不对、引用对不对、有没有泄漏思考过程。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8098"

CASES = [
    ("规则", "会员卡能跨店用吗", "跨店使用"),
    ("规则", "满减和折扣券能一起用吗", "叠加规则"),
    ("规则", "储值2000送多少钱", "储值与赠送"),
    ("规则", "客人说酒有异味怎么处理", "酒水异味"),
    ("规则", "什么情况必须叫店长", "升级与上报"),
    ("规则", "新入职第一天要带什么材料", "第一天"),
    ("规则", "探店视频里能说买一送一吗", "内容红线"),
    ("数据", "上周广州天河店卖了多少", None),
    ("数据", "广州天河店库存告急的有哪些", None),
    ("数据", "深圳福田店热销排行前五", None),
    ("拒答", "员工的工资是多少", None),
]

DIRTY = ("用户问的是", "我需要根据", "综上", "回顾", "让我们", "分析如下", "回答结构", "检查是否")


def is_leak(text: str) -> list[str]:
    """判断思考痕迹泄漏。

    注意不能把「首先」本身当泄漏信号：
    「首先确认是原料问题还是出品问题」是正常的中文答案开头。
    真正的痕迹是「首先，用户问的是…」这种复述问题的开场白。
    """
    hits = [w for w in DIRTY if w in text]
    if re.match(r"^首先[，,]\s*用户", text):
        hits.append("首先+复述")
    return hits


def main() -> None:
    out: list[str] = []
    thread = "verify-1"
    with httpx.Client(timeout=180.0) as c:
        h = c.get(f"{BASE}/health").json()
        out.append(f"health: embed={h['embed_backend']} llm={h['llm_backend']} "
                   f"db={h.get('db_backend')}/{h.get('ops_backend')} engine={h['engine']} "
                   f"chunks={h['stats']['chunks']}")
        out.append("")

        for kind, q, expect in CASES:
            t0 = time.perf_counter()
            r = c.post(f"{BASE}/chat", json={"query": q, "thread_id": thread})
            ms = int((time.perf_counter() - t0) * 1000)
            d = r.json()
            ans = (d.get("answer") or "").strip()
            cites = d.get("citations") or []
            top = cites[0]["section_path"] if cites else "-"
            tools = d.get("data") or []

            out.append("-" * 72)
            out.append(f"[{kind}] {q}    ({ms}ms)")
            out.append(f"  intent={d['intent']}  degraded={d['degraded']}  路由={d['route_reason']}")
            if expect:
                out.append(f"  期望章节含「{expect}」 → {'✔' if expect in top else '✘ 实际: ' + top}")
            out.append(f"  答案({len(ans)}字): {ans}")
            dirty = is_leak(ans)
            out.append(f"  思考泄漏: {'✘ ' + ','.join(dirty) if dirty else '✔ 无'}")
            if tools:
                for t in tools:
                    out.append(f"  工具 {t['tool']} ok={t['ok']}: {json.dumps(t['data'], ensure_ascii=False)}")
            gen = [s for s in d.get("trace", []) if s["step"] == "generate"]
            if gen:
                out.append(f"  generate: {gen[0]['elapsed_ms']}ms  {gen[0]['detail']}")
            out.append("")

    text = "\n".join(out)
    Path("data/verify_e2e.txt").write_text(text, encoding="utf-8")
    print("written data/verify_e2e.txt")


if __name__ == "__main__":
    main()





