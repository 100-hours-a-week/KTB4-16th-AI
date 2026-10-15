from pathlib import Path

import pytest

from app.components.axis_vectors import AxisVectors
from app.components.category_vectors import CategoryVectors
from app.components.clip_tagger import ClipTagger

FIXTURE_YAML = Path(__file__).parent / "fixtures" / "mood_tags_sample.yaml"


class ScriptedClipClient:
    """이미지·태그 벡터를 원하는 값으로 고정해 임계치·정렬 로직을 검증한다."""

    model = "scripted-clip"
    dim = 3

    def __init__(self, image_vector: list[float], tag_vectors: dict[str, list[float]]):
        self._image_vector = image_vector
        self._tag_vectors = tag_vectors

    async def encode_image(self, image_url: str) -> list[float]:
        return self._image_vector

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [self._tag_vectors[t] for t in texts]


def _tagger(image_vector, tag_vectors) -> ClipTagger:
    client = ScriptedClipClient(image_vector, tag_vectors)
    return ClipTagger(client, CategoryVectors(client, path=FIXTURE_YAML))


async def test_tag_returns_only_labels_above_threshold():
    tagger = _tagger(
        image_vector=[1.0, 0.0, 0.0],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 1.0, 0.0]},
    )

    tags = await tagger.tag("https://img/1.jpg")

    assert tags == ["노을"]  # night street는 직교(유사도 0)라 임계치 미달


async def test_tag_returns_top_k_ordered_by_score():
    tagger = _tagger(
        image_vector=[1.0, 1.0, 0.0],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.5, 0.5, 0.0]},
    )

    tags = await tagger.tag("https://img/1.jpg", top_k=2)

    assert tags == ["밤거리", "노을"]  # night street(밤거리) 벡터가 이미지에 더 가까움


async def test_tag_loads_categories_lazily_once():
    client = ScriptedClipClient(
        image_vector=[1.0, 0.0, 0.0],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 1.0, 0.0]},
    )
    categories = CategoryVectors(client, path=FIXTURE_YAML)
    tagger = ClipTagger(client, categories)

    assert not categories.loaded
    await tagger.tag("https://img/1.jpg")
    assert categories.loaded

    await tagger.tag("https://img/2.jpg")  # 두 번째 호출도 재사용만 하고 재로딩 안 함
    assert len(categories.tags) == 2


async def test_tag_uses_top_label_when_nothing_clears_threshold():
    """만화 짤·셀카처럼 장소 태그가 전부 미달이어도 사진 추천이 통째로 실패하지 않게 1등은 쓴다."""
    tagger = _tagger(
        image_vector=[0.1, 1.0, 0.0],  # sunset 유사도 ≈ 0.0995, night street 0 — 둘 다 임계치 미달
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 0.0, 1.0]},
    )

    assert await tagger.tag("https://img/1.jpg") == ["노을"]


async def test_tag_from_vector_stays_empty_when_nothing_clears_threshold():
    """기능2 RECAP 사진 분류는 억지로 붙이지 않는다 (분포가 왜곡되지 않게)."""
    tagger = _tagger(
        image_vector=[0.0, 0.0, 1.0],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 1.0, 0.0]},
    )

    assert await tagger.tag_from_vector([0.0, 0.0, 1.0]) == []


async def test_tag_from_vector_uses_given_vector_without_calling_clip():
    """기능2 RECAP이 기능4가 저장해둔 사진 벡터로 장면을 분류할 때 쓴다."""

    class NoImageClient(ScriptedClipClient):
        async def encode_image(self, image_url: str) -> list[float]:
            raise AssertionError("저장된 벡터가 있으면 CLIP 이미지 인코딩을 부르면 안 된다")

    client = NoImageClient(
        image_vector=[],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 1.0, 0.0]},
    )
    tagger = ClipTagger(client, CategoryVectors(client, path=FIXTURE_YAML))

    assert await tagger.tag_from_vector([0.0, 1.0, 0.0]) == ["밤거리"]


# --- 다축 태그 -----------------------------------------------------------------

AXIS_YAML = Path(__file__).parent / "fixtures" / "axis_tags_sample.yaml"

# 벡터 6칸: [진짜사진, 그림, 음식, 풍경, 웃음, 슬픔]. 문장마다 한 칸씩 가리키게 고정한다.
AXIS_TEXT_VECTORS = {
    "real photo": [1.0, 0, 0, 0, 0, 0],
    "camera photo": [1.0, 0, 0, 0, 0, 0],
    "cartoon": [0, 1.0, 0, 0, 0, 0],
    "food": [0, 0, 1.0, 0, 0, 0],
    "landscape": [0, 0, 0, 1.0, 0, 0],
    "laugh": [0, 0, 0, 0, 1.0, 0],
    "cry": [0, 0, 0, 0, 0, 1.0],
}


class AxisClipClient:
    model = "axis-clip"
    dim = 6

    def __init__(self):
        self.text_calls = 0

    async def encode_image(self, image_url: str) -> list[float]:
        raise AssertionError("저장된 벡터로 분류할 때 CLIP 이미지 인코딩을 부르면 안 된다")

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        self.text_calls += 1
        return [AXIS_TEXT_VECTORS[t] for t in texts]


def _axis_tagger() -> tuple[ClipTagger, AxisClipClient]:
    client = AxisClipClient()
    tagger = ClipTagger(
        client,
        CategoryVectors(client, path=FIXTURE_YAML),
        AxisVectors(client, path=AXIS_YAML),
    )
    return tagger, client


async def test_tag_from_vector_other_axis_gives_closest_tag():
    tagger, _ = _axis_tagger()

    assert await tagger.tag_from_vector([1.0, 0, 1.0, 0, 0, 0], axis="subject") == ["음식"]


async def test_tag_from_vector_other_axis_is_empty_when_uncertain():
    tagger, _ = _axis_tagger()

    # 음식과 풍경이 똑같이 가까워서 1등-2등 차이가 margin(0.1)보다 작다
    assert await tagger.tag_from_vector([0, 0, 1.0, 1.0, 0, 0], axis="subject") == []


async def test_tag_from_vector_default_axis_is_still_the_scene_axis():
    """axis를 안 주면 예전과 똑같이 장소 태그를 임계치로 거른다 (RECAP 차트·기능3이 쓰는 경로)."""

    class SceneClient(AxisClipClient):
        async def encode_text(self, texts: list[str]) -> list[list[float]]:
            vectors = {"sunset": [1.0, 0, 0, 0, 0, 0], "night street": [0, 1.0, 0, 0, 0, 0]}
            return [vectors[t] for t in texts]

    client = SceneClient()
    tagger = ClipTagger(client, CategoryVectors(client, path=FIXTURE_YAML))  # 축 벡터 없이도 동작

    assert await tagger.tag_from_vector([0, 1.0, 0, 0, 0, 0]) == ["밤거리"]


async def test_other_axis_without_axis_vectors_raises():
    client = AxisClipClient()
    tagger = ClipTagger(client, CategoryVectors(client, path=FIXTURE_YAML))

    with pytest.raises(ValueError):
        await tagger.tag_from_vector([1.0, 0, 0, 0, 0, 0], axis="subject")
    with pytest.raises(ValueError):
        await tagger.classify_axes([1.0, 0, 0, 0, 0, 0])


async def test_classify_axes_real_photo_gets_subject_not_expression():
    tagger, _ = _axis_tagger()

    result = await tagger.classify_axes([1.0, 0, 1.0, 0, 0, 0])

    assert result == {"사진 종류": "진짜 사진", "대상": "음식"}


async def test_classify_axes_meme_gets_expression_not_subject():
    tagger, _ = _axis_tagger()

    # 그림 + 웃음 방향. 음식·풍경 쪽 값은 0이라 대상 축은 아예 보지 않는다
    result = await tagger.classify_axes([0, 1.0, 0, 0, 1.0, 0])

    assert result == {"사진 종류": "그림·짤", "표정": "웃음"}


async def test_classify_axes_is_empty_when_photo_kind_is_uncertain():
    tagger, _ = _axis_tagger()

    # 진짜 사진과 그림이 똑같이 가까우면 어느 쪽 축을 볼지 모르므로 아무것도 붙이지 않는다
    assert await tagger.classify_axes([1.0, 1.0, 1.0, 0, 0, 0]) == {}


async def test_classify_axes_skips_only_the_uncertain_axis():
    tagger, _ = _axis_tagger()

    # 진짜 사진은 분명하지만 음식과 풍경이 똑같이 가까움 → 대상만 빠지고 사진 종류는 남는다
    result = await tagger.classify_axes([1.0, 0, 1.0, 1.0, 0, 0])

    assert result == {"사진 종류": "진짜 사진"}


async def test_axis_vectors_load_once_across_calls():
    tagger, client = _axis_tagger()

    await tagger.classify_axes([1.0, 0, 1.0, 0, 0, 0])
    await tagger.classify_axes([0, 1.0, 0, 0, 1.0, 0])
    await tagger.tag_from_vector([1.0, 0, 1.0, 0, 0, 0], axis="kind")

    assert client.text_calls == 1
