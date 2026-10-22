"""检索层的共享类型与后端工厂。

运行时只有 PostgreSQL + pgvector 一个后端（`pg_store.py`）。
这里保留 `tokenize` / `Hit`，因为 BM25 打分和上层 Pipeline 都依赖它们；
SQLite 实现已经移除，不再作为 fallback。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文取字符 bigram + 单字，英文数字取整词。"""
    tokens: list[str] = []
    for w in EN_RE.findall(text.lower()):
        tokens.append(w)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    doc_title: str
    section_path: str
    text: str
    score: float
    source: str = ""


def build_store(settings):
    """存储只有一个后端：PostgreSQL + pgvector。"""
    from .pg_store import PgStore

    return PgStore(settings.pg_dsn, dim=settings.embed_dim)
