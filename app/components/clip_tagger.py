"""CLIP 인코딩·zero-shot 태깅."""

from app.clients.clip_client import ClipClient


class ClipTagger:
    def __init__(self, client: ClipClient):
        self._client = client

    @property
    def model(self) -> str:
        return self._client.model

    async def encode_image(self, image_url: str) -> list[float]:
        return await self._client.encode_image(image_url)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return await self._client.encode_text(texts)

    async def tag(self, image_url: str, top_k: int = 3) -> list[str]:
        """이미지 벡터와 무드 라벨 벡터의 유사도로 태그 산출 — 사진 기반 추천 구현 시 작성."""
        raise NotImplementedError
