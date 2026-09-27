import pytest

from app.components.query_rewriter import QueryRewriter
from app.exceptions import LLMError


class ScriptedLLM:
    model = "scripted-llm"

    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.last_prompt = prompt
        return self._response


async def test_rewrite_from_tags_returns_cleaned_query():
    llm = ScriptedLLM('"노을 시티팝"\n')
    rewriter = QueryRewriter(llm)

    query = await rewriter.rewrite_from_tags(["노을", "따뜻한색"])

    assert query == "노을 시티팝"
    assert "노을, 따뜻한색" in llm.last_prompt


async def test_rewrite_from_tags_takes_first_line_only():
    llm = ScriptedLLM("잔잔한 발라드\n(참고: 태그 기반 추정)")
    query = await QueryRewriter(llm).rewrite_from_tags(["카페"])
    assert query == "잔잔한 발라드"


async def test_rewrite_from_tags_empty_tags_raises():
    with pytest.raises(LLMError):
        await QueryRewriter(ScriptedLLM("아무거나")).rewrite_from_tags([])
