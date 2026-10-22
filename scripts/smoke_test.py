"""端到端冒烟测试：不依赖端口，用 TestClient 直接打接口。

跑通这个，说明「接口 → 编排 → 检索/工具 → 存储」全链路是活的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

CASES = [
    ("policy", "会员卡能跨店用吗", "跨店使用"),
    ("policy", "客人说酒有异味怎么办", "酒水异味"),
    ("policy", "满减和折扣券能一起用吗", "叠加规则"),
    ("data", "上周广州天河店卖了多少", None),
    ("data", "广州天河店库存告急的有哪些", None),
]


def main() -> None:
    c = TestClient(app)
    out: list[str] = []

    r = c.get("/health")
    out.append(f"GET  /health  -> {r.status_code}  {json.dumps(r.json(), ensure_ascii=False)}")

    r = c.post("/ingest", json={"corpus_dir": "corpus"})
    body = r.json()
    out.append(f"POST /ingest  -> {r.status_code}  docs={body['indexed_docs']} "
               f"chunks={body['indexed_chunks']} skipped={body['skipped_unchanged']}")

    r = c.get("/docs/list")
    out.append(f"GET  /docs/list -> {r.status_code}  {len(r.json()['docs'])} 篇")

    passed = 0
    for kind, q, expect in CASES:
        r = c.post("/chat", json={"query": q})
        d = r.json()
        ok = r.status_code == 200 and bool(d["answer"])
        hit = True
        if expect:
            top = d["citations"][0]["section_path"] if d["citations"] else ""
            hit = expect in top
        if ok and hit:
            passed += 1
        out.append("")
        out.append("-" * 70)
        out.append(f"[{kind}] {q}")
        out.append(f"  status={r.status_code} engine={d.get('engine')} intent={d.get('intent')} conf={d.get('confidence')}")
        out.append(f"  route: {d.get('route_reason')}")
        for t in d.get("trace", []):
            out.append(f"  trace: {t['step']:<12} {t['elapsed_ms']:>4}ms  {t['detail']}")
        if d.get("citations"):
            out.append(f"  top1: 《{d['citations'][0]['doc_title']}》 {d['citations'][0]['section_path']}"
                       f"  {'OK' if hit else 'MISS(期望 ' + str(expect) + ')'}")
        for tr in d.get("data", []):
            out.append(f"  tool: {tr['tool']} ok={tr['ok']} {tr['data']}")
        out.append("  answer:")
        for line in (d.get("answer") or "").splitlines()[:8]:
            out.append("    " + line)

    # 多轮：同一 thread_id 共享上下文，靠 checkpointer
    out.append("")
    out.append("=" * 70)
    out.append("多轮会话（thread_id 共享上下文）")
    c.delete("/session/smoke-multi")
    for q in ("会员卡能跨店用吗", "那退卡呢", "手续费多少"):
        d = c.post("/chat", json={"query": q, "thread_id": "smoke-multi"}).json()
        top = d["citations"][0]["section_path"] if d["citations"] else "-"
        out.append(f"  问: {q}")
        out.append(f"    检索 query={d.get('search_query')}  intent={d.get('intent')}  top1={top}")
    hist = c.get("/session/smoke-multi/history").json()
    out.append(f"  会话历史: {hist['turns']}")

    out.append("")
    out.append("=" * 70)
    out.append(f"命中正确段落: {passed}/{len(CASES)}")

    text = "\n".join(out)
    (ROOT / "data" / "smoke_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
