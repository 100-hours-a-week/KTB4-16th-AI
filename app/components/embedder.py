"""텍스트 → 벡터. 임베딩 모델 교체 시 client만 바꾼다."""

from app.clients.embedding_client import EmbeddingClient


class Embedder:
    def __init__(self, client: EmbeddingClient):
        self._client = client

    @property
    def model(self) -> str:
        return self._client.model

    async def embed(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("빈 텍스트는 임베딩할 수 없어요.")
        return (await self._client.embed([cleaned]))[0]
