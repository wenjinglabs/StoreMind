"""全局配置。所有东西都可插拔，靠环境变量切换，不写死。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """读项目根 .env，填进 os.environ。

    不引 python-dotenv：就这么几行，而且已经存在的环境变量优先，
    这样 CI / Docker 里注入的值不会被本地文件覆盖。
    """
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # 数据与语料
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    corpus_dir: Path = field(default_factory=lambda: ROOT / "corpus")

    # 嵌入后端: hashing | ollama
    # hashing 零依赖，用于无网络/未拉模型时跑通链路；ollama 是正式路径
    embed_backend: str = field(default_factory=lambda: _env("SM_EMBED_BACKEND", "hashing"))
    embed_model: str = field(default_factory=lambda: _env("SM_EMBED_MODEL", "bge-m3"))
    ollama_base: str = field(default_factory=lambda: _env("SM_OLLAMA_BASE", "http://172.16.0.71:11434"))

    # 生成后端: none | ollama | openai
    # none 时走抽取式兜底，保证没有 key 也能演示
    llm_backend: str = field(default_factory=lambda: _env("SM_LLM_BACKEND", "none"))
    llm_model: str = field(default_factory=lambda: _env("SM_LLM_MODEL", "qwen2.5:7b"))
    openai_base: str = field(default_factory=lambda: _env("SM_OPENAI_BASE", "https://api.openai.com/v1"))
    openai_key: str = field(default_factory=lambda: _env("SM_OPENAI_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("SM_OPENAI_MODEL", "gpt-4o-mini"))
    # 生成超时。UI 是同步等答案的，超时太长会让人以为卡死；
    # 超时后走抽取式兜底，宁可答案朴素也不要转圈。
    llm_timeout: float = field(default_factory=lambda: float(_env("SM_LLM_TIMEOUT", "25")))

    # 切分参数
    chunk_min_chars: int = 200
    chunk_max_chars: int = 450
    chunk_overlap_chars: int = 60

    # 检索参数
    vector_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    # 拒答阈值。默认 0 表示不启用，原因见 pipeline._node_retrieve 的注释：
    # 在 45 chunk 的规模上，命中组与无答案组的 BM25 分完全重叠（11.37 vs 11.67），
    # 任何阈值都是在编数字。拒答应该由生成模型做（语义判断），而不是拿检索分硬凑。
    min_relevance: float = 0.0

    # Agent 层：自检不通过时最多重查几轮。
    # 设成 0 就是关掉 Agent 层，退回纯流水线——用来做 A/B 对照，
    # 量化「自检重查」到底带来了多少收益（scripts/probe_agent_ab.py）。
    agent_max_attempts: int = field(
        default_factory=lambda: int(_env("SM_AGENT_MAX_ATTEMPTS", "2"))
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "storemind.sqlite3"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
