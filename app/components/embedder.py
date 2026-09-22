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

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 API 호출 1번으로 임베딩한다 (재랭킹 후보곡처럼 텍스트 수가
        많을 때, 하나씩 부르는 대신 배치로 묶어 호출 수를 줄인다)."""
        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("빈 텍스트는 임베딩할 수 없어요.")
        if not cleaned:
            return []
        return await self._client.embed(cleaned)
