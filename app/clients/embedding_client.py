"""텍스트 임베딩 API 어댑터."""

from typing import Protocol

import openai

from app.clients.fake import fake_vector
from app.exceptions import EmbeddingError


class EmbeddingClient(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, model: str, dim: int, timeout: float, max_retries: int):
        self.model = model
        self.dim = dim
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            res = await self._client.embeddings.create(model=self.model, input=texts)
        except openai.OpenAIError as e:
            raise EmbeddingError(f"임베딩 API 호출 실패: {type(e).__name__}") from e
        vectors = [d.embedding for d in sorted(res.data, key=lambda d: d.index)]
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"임베딩 차원이 설정값({self.dim})과 달라요.")
        return vectors


class FakeEmbeddingClient:
    def __init__(self, dim: int, model: str = "fake-text-embedding"):
        self.model = model
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"text:{t}", self.dim) for t in texts]
