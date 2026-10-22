"""探测 opencode zen 的实际端点与可用模型。

文档站有多个版本：/zen/v1 与 /inference/openai/v1 都出现过，
所以不猜，直接打表试。key 从 .env 读，不打印。

用法: python scripts/probe_llm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


BASES = [
    "https://opencode.ai/zen/v1",
    "https://opencode.ai/inference/openai/v1",
]

MODELS = [
    "deepseek-v4.1-flash-free",
    "deepseek-v4.1-flash",
    "deepseek-v4.1-flash:free",
]

CHAT_MODELS = [
    "deepseek-v4.1-flash-free",
    "glm-5.3-flash-free",
    "big-pickle",
]


def main() -> None:
    env = load_env()
    key = env.get("SM_OPENAI_KEY", "")
    lines: list[str] = []
    if not key:
        lines.append("!! .env 里没有 SM_OPENAI_KEY")
        (ROOT / "data" / "_probe_llm.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return
    lines.append(f"key 前缀: {key[:8]}... 长度={len(key)}")

    with httpx.Client(timeout=40.0) as c:
        for base in BASES:
            lines.append("")
            lines.append(f"== GET {base}/models ==")
            try:
                r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                lines.append(f"  status={r.status_code}")
                if r.status_code == 200:
                    data = r.json()
                    ids = [m.get("id") for m in data.get("data", [])]
                    lines.append(f"  模型数={len(ids)}")
                    free = [i for i in ids if i and ("free" in i or "flash" in i or "pickle" in i)]
                    lines.append(f"  含 free/flash 的: {free[:20]}")
                    ds = [i for i in ids if i and "deepseek" in i.lower()]
                    lines.append(f"  deepseek 系列: {ds}")
                else:
                    lines.append(f"  body={r.text[:200]}")
            except Exception as e:
                lines.append(f"  ERR {type(e).__name__}: {e}")

        for base in BASES:
            for model in CHAT_MODELS:
                lines.append("")
                lines.append(f"== POST {base}/chat/completions model={model} ==")
                try:
                    r = c.post(
                        f"{base}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "回答两个字：收到"}],
                            "max_tokens": 32,
                        },
                    )
                    lines.append(f"  status={r.status_code}")
                    if r.status_code == 200:
                        d = r.json()
                        msg = d["choices"][0]["message"]["content"]
                        lines.append(f"  OK content={msg!r}")
                        lines.append(f"  usage={d.get('usage')}")
                    else:
                        lines.append(f"  body={r.text[:300]}")
                except Exception as e:
                    lines.append(f"  ERR {type(e).__name__}: {e}")

    text = "\n".join(lines)
    (ROOT / "data" / "_probe_llm.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
