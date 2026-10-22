"""MCP server 的启动入口。

为什么不直接用 `python -m app.mcp.server`：MCP 客户端（Claude Desktop / Cursor 等）
拉起子进程时工作目录不一定在项目根，`-m app...` 会 import 失败。
这个文件把项目根塞进 sys.path，客户端只要指向这个文件的绝对路径就能跑。

    python scripts/mcp_server.py --stdio
    python scripts/mcp_server.py --http 8765
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.mcp.server import main  # noqa: E402 - 必须先改 sys.path

if __name__ == "__main__":
    raise SystemExit(main())
