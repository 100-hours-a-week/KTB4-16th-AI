"""LLM 인터페이스 — 구현체 교체 시 이 계약만 지키면 된다."""

from typing import Protocol


class LLMClient(Protocol):
    model: str

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class LocalLLMClient:
    """RunPod에서 직접 서빙하는 로컬 LLM — V2 구현 예정."""

    def __init__(self, url: str, model: str):
        self.model = model or "local-llm"
        self._url = url

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError("로컬 LLM은 V2에서 구현 (LLM_*_PROVIDER=anthropic 사용)")


class FakeLLMClient:
    model = "fake-llm"

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return f"[fake] {prompt[:80]} 에 어울리는 잔잔하고 따뜻한 분위기. 태그: 잔잔함, 따뜻함"
