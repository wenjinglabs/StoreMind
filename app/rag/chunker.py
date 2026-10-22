"""按 Markdown 标题层级做语义切分。

为什么不按固定 token 切：门店 SOP 的知识单元是「章节」，一个章节讲一件事。
按 512 token 硬切会把「跨店核销的三种情况」切成两半，检索出来的都是残句。

核心技巧：embedding 时把「文档标题 / 章节路径」拼在正文前面。
标题是极强的人工语义标签，白扔掉可惜。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENT_END = "。！？；!?;\n"


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section_path: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    @property
    def embed_text(self) -> str:
        """送进嵌入模型的文本：带上标题路径。"""
        return f"{self.doc_title} / {self.section_path}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in SENT_END:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [s for s in out if s.strip()]


def _pack(paragraphs: list[str], min_chars: int, max_chars: int) -> list[str]:
    """把段落打包成 200~450 字的块。超长段落先按句子拆。"""
    units: list[str] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            units.append(p)
        else:
            for s in _split_sentences(p):
                units.append(s.strip())

    blocks: list[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if cur and len(cur) + len(u) + 1 > max_chars:
            blocks.append(cur.strip())
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur.strip():
        blocks.append(cur.strip())

    # 尾部过短的块并入前一块，避免碎块稀释检索
    if len(blocks) >= 2 and len(blocks[-1]) < min_chars // 2:
        tail = blocks.pop()
        blocks[-1] = blocks[-1] + "\n" + tail
    return blocks


def _tail_sentences(text: str, max_chars: int) -> str:
    """取末尾若干净句，用于 overlap。

    第一版是直接切最后 N 个字符，结果每个 chunk 开头都是半句话，
    生成时模型会读到错位的上文。改成按完整句子从后往前取。
    """
    if max_chars <= 0:
        return ""
    sents = _split_sentences(text)
    out: list[str] = []
    total = 0
    for s in reversed(sents):
        if total + len(s) > max_chars and out:
            break
        out.append(s)
        total += len(s)
    return "".join(reversed(out)).lstrip()


def split_markdown(
    raw: str,
    doc_id: str,
    doc_title: str,
    min_chars: int = 200,
    max_chars: int = 450,
    overlap_chars: int = 60,
) -> list[Chunk]:
    lines = raw.splitlines()
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str, int]] = []
    buf: list[str] = []
    offset = 0
    sec_start = 0

    def flush() -> None:
        nonlocal buf, sec_start
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            path = " > ".join(t for _, t in heading_stack) or doc_title
            sections.append((path, text, sec_start))
        buf = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            sec_start = offset
        else:
            if not buf:
                sec_start = offset
            buf.append(line)
        offset += len(line) + 1
    flush()

    chunks: list[Chunk] = []
    idx = 0
    prev_tail = ""
    prev_section = ""
    for path, text, start in sections:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        for block in _pack(paragraphs, min_chars, max_chars):
            # 只在同一章节内的连续块之间做 overlap。
            # 跨章节 overlap 会把上一章的内容贴到下一章开头，污染检索和生成。
            body = f"{prev_tail}{block}" if (prev_tail and path == prev_section) else block
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=path,
                    chunk_index=idx,
                    text=body,
                    char_start=start,
                    char_end=start + len(body),
                )
            )
            idx += 1
            prev_tail = _tail_sentences(block, overlap_chars)
            prev_section = path
    return chunks
