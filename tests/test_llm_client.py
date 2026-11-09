import pytest

from app.clients.llm_client import FallbackLLMClient
from app.exceptions import LLMError


class StubLLM:
    def __init__(self, model, answer=None, error=None):
        self.model = model
        self._answer = answer
        self._error = error
        self.calls = []

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        json_mode: bool = False,
        image_url: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "image_url": image_url,
            }
        )
        if self._error:
            raise self._error
        return self._answer


async def test_fallback_uses_the_primary_and_never_calls_the_fallback_when_it_works():
    primary = StubLLM("gemma", answer="젬마 글")
    fallback = StubLLM("haiku", answer="하이쿠 글")

    text = await FallbackLLMClient(primary, fallback).complete(system="s", prompt="p")

    assert text == "젬마 글" and (len(primary.calls), len(fallback.calls)) == (1, 0)


async def test_fallback_switches_to_claude_when_the_primary_raises_llm_error(caplog):
    primary = StubLLM("gemma", error=LLMError("로컬 LLM 연결 실패"))
    fallback = StubLLM("haiku", answer="하이쿠 글")
    client = FallbackLLMClient(primary, fallback)

    with caplog.at_level("WARNING", logger="muro.llm"):
        text = await client.complete(system="s", prompt="p")

    assert text == "하이쿠 글" and (len(primary.calls), len(fallback.calls)) == (1, 1)
    assert client.model == "gemma"  # 이름은 기본(primary) 쪽
    assert "gemma" in caplog.text and "haiku" in caplog.text  # 어느 쪽으로 넘겼는지 로그에 남는다


async def test_fallback_passes_every_argument_to_whichever_client_answers():
    primary = StubLLM("gemma", error=LLMError("서버 꺼짐"))
    fallback = StubLLM("haiku", answer="하이쿠 글")

    await FallbackLLMClient(primary, fallback).complete(
        system="시스템", prompt="입력", max_tokens=300, json_mode=True, image_url="https://x/y.jpg"
    )

    expected = {
        "system": "시스템",
        "prompt": "입력",
        "max_tokens": 300,
        "json_mode": True,
        "image_url": "https://x/y.jpg",
    }
    assert primary.calls == [expected] and fallback.calls == [expected]


async def test_fallback_error_propagates_when_both_fail():
    primary = StubLLM("gemma", error=LLMError("서버 꺼짐"))
    fallback = StubLLM("haiku", error=LLMError("Claude 장애"))

    with pytest.raises(LLMError, match="Claude 장애"):
        await FallbackLLMClient(primary, fallback).complete(system="s", prompt="p")


async def test_fallback_does_not_hide_unexpected_errors_from_the_primary():
    primary = StubLLM("gemma", error=KeyError("버그"))
    fallback = StubLLM("haiku", answer="하이쿠 글")

    with pytest.raises(KeyError):
        await FallbackLLMClient(primary, fallback).complete(system="s", prompt="p")
    assert fallback.calls == []
