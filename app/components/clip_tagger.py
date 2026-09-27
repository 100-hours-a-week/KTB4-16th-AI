"""CLIP 인코딩·zero-shot 태깅."""

from app.clients.clip_client import ClipClient
from app.components.category_vectors import CategoryVectors
from app.components.vector_math import cosine_similarity

# 실측 기준(영어 태그, 사무실 야근 사진): 1위 20.8%, 2위 16.1%, 3위 14.6%,
# 노이즈는 8~13%대. TOP_K는 넉넉히 잡고 THRESHOLD로 노이즈를 거른다.
TOP_K = 3
THRESHOLD = 0.15


class ClipTagger:
    def __init__(self, client: ClipClient, categories: CategoryVectors):
        self._client = client
        self._categories = categories

    @property
    def model(self) -> str:
        return self._client.model

    async def encode_image(self, image_url: str) -> list[float]:
        return await self._client.encode_image(image_url)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return await self._client.encode_text(texts)

    async def tag(self, image_url: str, top_k: int = TOP_K) -> list[str]:
        """이미지 벡터와 무드 라벨 벡터의 유사도로 태그(한국어 표시명) 산출.

        임계치 미달 후보는 제외 — 사진과 무관한 태그가 섞여 들어가는 것을
        막는다. 전부 미달이면 빈 리스트(호출부가 폴백 처리).
        """
        image_vector = await self._client.encode_image(image_url)
        return await self.tag_from_vector(image_vector, top_k)

    async def tag_from_vector(self, image_vector: list[float], top_k: int = TOP_K) -> list[str]:
        """이미 만들어둔 CLIP 이미지 벡터로 태깅 — CLIP을 다시 부르지 않는다.

        기능2 RECAP이 기능4가 저장해둔 사진 벡터(record_embeddings.image_embedding)로
        사진 장면을 분류할 때 쓴다.
        """
        if not self._categories.loaded:
            await self._categories.load()

        scored = [
            (tag.ko, cosine_similarity(image_vector, tag.vector)) for tag in self._categories.tags
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [ko for ko, score in scored[:top_k] if score >= THRESHOLD]
