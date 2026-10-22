"""嵌入层：一个接口，两个实现。

hashing  —— 零依赖、确定性、离线可用。用于跑通链路和 CI，不追求效果上限。
ollama   —— 正式路径，bge-m3（中文强、dense+sparse 一体）。

这样设计的原因：RAG 的调试需求是「每一步能单独看」，
如果第一步就卡在拉 2GB 模型上，你根本没法验证后面的切片和检索对不对。
"""
from __future__ import annotations

import hashlib
import math
import zlib
from abc import ABC, abstractmethod

import httpx


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def signature(self) -> str:
        """标识「这批向量是用什么算出来的」。

        索引里存这个值，用来检测嵌入器是否被换过。
        换后端不重建索引是最阴险的坑：维度一变，旧向量会被
        `len(v) != len(q)` 静默跳过，表现成「召回变差」而不是「配置错了」。

        只用「后端 + 模型名」，不带 dim：ollama 的维度要等首次嵌入后才知道，
        带进签名会导致每次运行都判定为「变了」而误触发重建。
        hashing 那边已经把维度编码进 model 名里了。
        """
        return f"{self.name}:{getattr(self, 'model', '')}"


def _ngrams(text: str, ns: tuple[int, ...] = (2, 3)) -> list[str]:
    s = "".join(ch for ch in text if not ch.isspace())
    out: list[str] = []
    for n in ns:
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            out.append(s[i : i + n])
    return out


class HashingEmbedder(Embedder):
    """字符 n-gram 哈希投影 + L2 归一化。

    不是「假嵌入」，它是 subword hashing 的一种退化形式，
    对中文短文本的语义近邻有一定捕捉能力（共享 2/3-gram 越多越相似）。
    项目演示和回归测试够用，正式效果请切 ollama。
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model = f"ngram{hashlib.sha1(str(dim).encode()).hexdigest()[:6]}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        grams = _ngrams(text)
        if not grams:
            return vec
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        for g, c in counts.items():
            h = zlib.crc32(g.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0  # 带符号哈希，减少碰撞偏置
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, base: str, model: str, timeout: float = 60.0) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # 新版 /api/embed 支持批量。
        # keep_alive 显式给足：默认 5 分钟后模型被卸载，
        # 下次请求要重新加载 0.64GB 权重，延迟从十几毫秒跳到秒级。
        # 嵌入是 RAG 里每问必做的一步，秒级延迟会直接毁掉体验。
        payload = {
            "model": self.model,
            "input": texts,
            "keep_alive": "30m",
        }
        try:
            r = self._client.post(f"{self.base}/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
            if "embeddings" in data:
                self.dim = len(data["embeddings"][0])
                return data["embeddings"]
        except Exception:
            pass
        # 旧版 /api/embeddings 只收单条
        out: list[list[float]] = []
        for t in texts:
            r = self._client.post(
                f"{self.base}/api/embeddings",
                json={"model": self.model, "prompt": t, "keep_alive": "30m"},
            )
            r.raise_for_status()
            vec = r.json()["embedding"]
            self.dim = len(vec)
            out.append(vec)
        return out


def build_embedder(backend: str, *, ollama_base: str, model: str) -> Embedder:
    backend = (backend or "hashing").lower()
    if backend == "ollama":
        return OllamaEmbedder(ollama_base, model)
    return HashingEmbedder()


def fingerprint(text: str) -> str:
    """内容指纹，用于增量索引时判断是否需要重新嵌入。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
