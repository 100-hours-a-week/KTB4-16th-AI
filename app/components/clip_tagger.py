"""CLIP 인코딩·zero-shot 태깅."""

import logging
from dataclasses import dataclass, field

from app.clients.clip_client import ClipClient
from app.components.axis_vectors import AxisVectors, pick_tag
from app.components.category_vectors import CategoryVectors
from app.components.vector_math import cosine_similarity
from app.exceptions import ClipError

logger = logging.getLogger("muro.clip_tagger")

# 실측 기준(영어 태그, 사무실 야근 사진): 1위 20.8%, 2위 16.1%, 3위 14.6%,
# 노이즈는 8~13%대. TOP_K는 넉넉히 잡고 THRESHOLD로 노이즈를 거른다.
TOP_K = 3
THRESHOLD = 0.15

# 장소 축은 mood_tags.yaml(CategoryVectors)에 있고, 나머지 축은 axis_tags.yaml(AxisVectors)에 있다
SCENE_AXIS = "scene"


@dataclass(frozen=True)
class PhotoAnalysis:
    """사진 1장의 분석 결과. 장소 태그와 다축 태그(축 이름 → 태그)."""

    scene_tags: list[str]
    axes: dict[str, str] = field(default_factory=dict)


class ClipTagger:
    def __init__(
        self,
        client: ClipClient,
        categories: CategoryVectors,
        axes: AxisVectors | None = None,
    ):
        self._client = client
        self._categories = categories
        self._axes = axes

    @property
    def model(self) -> str:
        return self._client.model

    async def encode_image(self, image_url: str) -> list[float]:
        return await self._client.encode_image(image_url)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return await self._client.encode_text(texts)

    async def tag(self, image_url: str, top_k: int = TOP_K) -> list[str]:
        """기능3 사진 추천용 태깅 — 이미지와 무드 라벨 벡터의 유사도로 태그(한국어 표시명) 산출.

        임계치 넘는 태그를 쓰되, 하나도 없으면 1등 태그 하나는 쓴다. 태그가 전부 장소·풍경이라
        만화 짤·셀카·음식 사진은 전부 미달이 되는데(실측: 픽셀 만화 짤 1등 콘서트 0.1497),
        그때 추천이 통째로 실패하는 것보다 가장 가까운 분위기로라도 추천하는 게 낫다.
        """
        image_vector = await self._client.encode_image(image_url)
        return await self.tag_from_vector(image_vector, top_k, at_least_one=True)

    async def analyze(self, image_url: str, top_k: int = TOP_K) -> PhotoAnalysis:
        """기능3 사진 추천용 — 사진 벡터를 한 번만 만들어 장소 태그와 다축 태그를 함께 얻는다.

        장소 태그는 tag()와 같은 방식이라 실패(ClipError)도 똑같이 올려보낸다. 다축 태그는 곡을
        더 잘 고르기 위한 부가 정보라서, 분류하지 못해도(축 벡터가 없거나 태그 문장 인코딩이
        실패) 장소 태그만으로 계속하도록 빈 dict를 돌려준다.
        """
        image_vector = await self._client.encode_image(image_url)
        scene_tags = await self.tag_from_vector(image_vector, top_k, at_least_one=True)
        if self._axes is None:
            return PhotoAnalysis(scene_tags=scene_tags)
        try:
            axes = await self.classify_axes(image_vector)
        except ClipError:
            logger.warning("다축 태그를 분류하지 못해 장소 태그만 써요.", exc_info=True)
            axes = {}
        return PhotoAnalysis(scene_tags=scene_tags, axes=axes)

    async def tag_from_vector(
        self,
        image_vector: list[float],
        top_k: int = TOP_K,
        *,
        at_least_one: bool = False,
        axis: str = SCENE_AXIS,
    ) -> list[str]:
        """이미 만들어둔 CLIP 이미지 벡터로 태깅 — CLIP을 다시 부르지 않는다.

        기능2 RECAP이 기능4가 저장해둔 사진 벡터(record_embeddings.image_embedding)로
        사진 장면을 분류할 때 쓴다.

        axis가 "scene"(기본)이면 장소 태그를 임계치로 거른다. 그 외 축(kind, subject, vibe,
        expression)은 1등과 2등의 차이(margin)로 판단해서 태그 하나, 불확실하면 빈 목록을 준다
        (이때 top_k와 at_least_one은 쓰지 않는다).
        """
        if axis != SCENE_AXIS:
            axes = await self._loaded_axes()
            picked = pick_tag(image_vector, axes.get(axis))
            return [picked] if picked else []

        if not self._categories.loaded:
            await self._categories.load()

        scored = [
            (tag.ko, cosine_similarity(image_vector, tag.vector)) for tag in self._categories.tags
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        tags = [ko for ko, score in scored[:top_k] if score >= THRESHOLD]
        if not tags and at_least_one and scored:
            return [scored[0][0]]
        return tags

    async def classify_axes(self, image_vector: list[float]) -> dict[str, str]:
        """사진 종류를 먼저 가르고, 그 결과에 맞는 축만 이어서 분류한다. 축 이름 → 태그.

        진짜 사진이면 대상·분위기를, 그림·짤이면 표정·감정을 본다. 축마다 불확실하면 그 축은
        결과에 넣지 않는다. 사진 종류부터 불확실하면 빈 dict다(어느 쪽 축을 볼지 모르므로).
        장소(scene)는 여기 없다 — tag_from_vector(axis="scene")로 따로 본다.
        """
        axes = await self._loaded_axes()
        kind = pick_tag(image_vector, axes.root)
        if kind is None:
            return {}
        result = {axes.root.ko: kind}
        for axis in axes.children_of(kind):
            if picked := pick_tag(image_vector, axis):
                result[axis.ko] = picked
        return result

    async def _loaded_axes(self) -> AxisVectors:
        if self._axes is None:
            raise ValueError("ClipTagger에 AxisVectors를 넣지 않아서 다축 태그를 쓸 수 없어요.")
        if not self._axes.loaded:
            await self._axes.load()
        return self._axes
