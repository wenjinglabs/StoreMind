"""诊断：答案引用的编号跟实际召回列表对不对得上。

上一轮验证发现「什么情况必须叫店长」的 top1 是
「新人入职第一周指南 > 第一周必须掌握的六件事」，但答案内容是
「升级与上报」里的四条情形。答案引用的是 [2]。

要确认的是：答案引用的 [2] 到底指向哪个章节？
如果 [2] 正是「升级与上报」，那答案没错，只是 top1 排序不理想；
如果 [2] 也是「第一周必须掌握的六件事」，那就是模型在编造引用。
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8091"
QUERIES = [
    "什么情况必须叫店长",
    "上周广州天河店卖了多少",
    "客人说酒有异味怎么处理",
]


def main() -> None:
    out: list[str] = []
    with httpx.Client(timeout=180.0) as c:
        for q in QUERIES:
            d = c.post(f"{BASE}/chat", json={"query": q, "thread_id": "diag"}).json()
            out.append("=" * 72)
            out.append(f"Q: {q}")
            out.append(f"答案: {d['answer']}")
            out.append("召回列表:")
            for ci in d.get("citations", []):
                out.append(f"   [{ci['index']}] {ci['section_path']}  score={ci['score']}")
            out.append("trace:")
            for t in d.get("trace", []):
                out.append(f"   {t['step']:<12} {t['elapsed_ms']:>5}ms  {t['detail']}")
            out.append("")
    text = "\n".join(out)
    Path("data/diag_cite.txt").write_text(text, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
