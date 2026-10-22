"""查一条语料到底被切成了哪几段，以及某个关键句落在哪一段。

背景：「客人吃到了坏了怎么办」的答案里少了「客人身体不适 → 建议就医」，
但这句话确实写在语料里。它在「酒水异味或品质问题」段落的最后一句，
怀疑是被切分时划到了下一段（属于「服务类投诉」章节）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.chunker import split_markdown
from app.rag.store import Store

settings.ensure_dirs()
store = Store(settings.db_path)

md = ROOT / "corpus" / "sop" / "03-客诉处理手册.md"
raw = md.read_text(encoding="utf-8")
chunks = split_markdown(
    raw, "sop/03-客诉处理手册.md", "客诉处理手册",
    min_chars=settings.chunk_min_chars,
    max_chars=settings.chunk_max_chars,
    overlap_chars=settings.chunk_overlap_chars,
)

out: list[str] = []
out.append(f"原文 {len(raw)} 字 → 切成 {len(chunks)} 段")
out.append(f"chunk 参数: {settings.chunk_min_chars}~{settings.chunk_max_chars} 字, overlap {settings.chunk_overlap_chars}")
out.append("")

KEY = "身体不适"
for c in chunks:
    has = KEY in c.text
    mark = "  ★ 命中关键句" if has else ""
    out.append(f"--- chunk {c.chunk_index}  [{c.section_path}]  {len(c.text)} 字{mark}")
    out.append(c.text.replace("\n", "\n    "))
    out.append("")

out.append("=" * 60)
out.append(f"包含「{KEY}」的段落索引: {[c.chunk_index for c in chunks if KEY in c.text]}")
out.append(f"包含「{KEY}」的章节: {[c.section_path for c in chunks if KEY in c.text]}")

text = "\n".join(out)
Path("data/chunk_dump.txt").write_text(text, encoding="utf-8")
print("written data/chunk_dump.txt")
