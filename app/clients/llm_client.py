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
        # music_mood_tagger.py의 "설명:/장르:/무드:" 형식에 맞춰둔다 — 이 클라이언트를
        # 쓰는 다른 컴포넌트(song_curator 등)는 이 형식에서 자기가 원하는 줄을 못 찾으면
        # 그냥 빈 결과로 처리하도록 이미 설계돼 있다(할루시네이션 방어와 같은 경로).
        return (
            f"설명: [fake] {prompt[:80]} 에 어울리는 잔잔하고 따뜻한 분위기\n"
            "장르: 발라드\n"
            "무드: 잔잔한,따뜻한"
        )
