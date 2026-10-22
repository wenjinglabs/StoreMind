"""生成层：可插拔，且必须支持「没有模型也能演示」。

none 后端返回 None，由 pipeline 走抽取式兜底。
这不是妥协，是工程要求：RAG 的检索对不对，不应该依赖你此刻有没有 API key。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx


class LLM(ABC):
    name = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        ...


class NoneLLM(LLM):
    name = "none"

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        return None


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, base: str, model: str, timeout: float = 25.0) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        """用 /api/chat 而不是 /api/generate。

        原因：/api/chat 支持 system 角色，我们的系统提示词需要独立字段，
        而 /api/generate 只能把 system 拼进 prompt。
        """
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        body: dict = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            # keep_alive 给足：默认 5 分钟后模型被卸载，
            # 下次请求要重新加载 2.5GB 权重，延迟从 0.4 秒跳到 17 秒。
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": 400},
        }
        # qwen3 是混合推理模型，think=true 时推理进 message.thinking、
        # content 只剩最终答案；think=false 时推理会直接混进 content，
        # 用户会看到「首先我需要...」这类思考过程。
        # 实测（scripts/probe_models.py）qwen3:4b 两种用法都不干净，
        # 所以默认用 qwen2.5 这类纯 instruct 模型，不传 think 字段。
        if "qwen3" in self.model.lower():
            body["think"] = False
        try:
            # think 字段是 qwen3 之后才有的，老版本收到未知字段会 400。
            # 只有真的带了 think 才需要回退重试。
            try:
                r = self._client.post(f"{self.base}/api/chat", json=body)
                r.raise_for_status()
            except httpx.HTTPStatusError:
                if "think" not in body:
                    raise
                body.pop("think", None)
                r = self._client.post(f"{self.base}/api/chat", json=body)
                r.raise_for_status()
            payload = r.json()
            msg = payload.get("message") or {}
            content = msg.get("content") or ""
            # 混合推理模型可能把回答放进 thinking 字段（content 为空），
            # 这时取 thinking——虽然通常是推理过程，但总比返回空强。
            if not content.strip():
                content = msg.get("thinking") or ""
            return content.strip() or None
        except Exception:
            return None


class OpenAICompatLLM(LLM):
    name = "openai"

    def __init__(self, base: str, key: str, model: str, timeout: float = 25.0) -> None:
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        if not self.key:
            return None
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            r = self._client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "messages": msgs, "temperature": temperature},
            )
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip() or None
        except Exception:
            return None


def build_llm(backend: str, **kw) -> LLM:
    backend = (backend or "none").lower()
    timeout = float(kw.get("timeout") or 25.0)
    if backend == "ollama":
        return OllamaLLM(kw["ollama_base"], kw["model"], timeout=timeout)
    if backend in ("openai", "deepseek"):
        return OpenAICompatLLM(kw["base"], kw["key"], kw["model"], timeout=timeout)
    return NoneLLM()
