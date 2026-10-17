"""텍스트 임베딩 API 어댑터."""

from typing import Protocol

import httpx
import openai

from app.clients.fake import fake_vector
from app.exceptions import EmbeddingError
from app.schemas.text_embed import MAX_TEXTS_PER_REQUEST


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


class LocalEmbeddingClient:
    """같은 컨테이너의 임베딩 서비스(embedding_main)를 부른다. 모델은 그쪽에 한 번만 올라간다.

    서비스가 아직 모델을 올리는 중이거나 죽어 있으면 EmbeddingError(→ 503)가 난다. 작업 큐(worker)는
    재시도하고, gateway 요청은 503을 돌려준다.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        model: str,
        dim: int,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.dim = dim
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Internal-Token": token},
            timeout=timeout,
            transport=transport,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
            vectors += await self._embed_chunk(texts[start : start + MAX_TEXTS_PER_REQUEST])
        return vectors

    async def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        try:
            res = await self._http.post("/api/embed", json={"texts": texts})
        except httpx.HTTPError as e:
            raise EmbeddingError(f"임베딩 서비스 연결 실패: {type(e).__name__}") from e
        if res.status_code != 200:
            raise EmbeddingError(f"임베딩 서비스 오류 (status {res.status_code})")
        try:
            vectors = res.json()["vectors"]
        except (ValueError, KeyError, TypeError) as e:
            raise EmbeddingError("임베딩 서비스 응답 형식이 달라요.") from e
        if len(vectors) != len(texts):
            raise EmbeddingError("임베딩 서비스가 보낸 벡터 개수가 글 개수와 달라요.")
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"임베딩 차원이 설정값({self.dim})과 달라요.")
        return vectors


class FakeEmbeddingClient:
    def __init__(self, dim: int, model: str = "fake-text-embedding"):
        self.model = model
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"text:{t}", self.dim) for t in texts]
