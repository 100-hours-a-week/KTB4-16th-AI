"""Claude API 어댑터."""

import anthropic

from app.exceptions import LLMError


class AnthropicClient:
    def __init__(self, api_key: str, model: str, timeout: float, max_retries: int):
        self.model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        try:
            res = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low"},
            )
        except anthropic.APIStatusError as e:
            raise LLMError(f"LLM API 오류 (status {e.status_code})") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("LLM API 연결 실패") from e

        if res.stop_reason == "refusal":
            raise LLMError("LLM이 요청을 거절했어요.")
        text = "".join(block.text for block in res.content if block.type == "text").strip()
        if not text:
            raise LLMError("LLM 응답이 비어 있어요.")
        return text
