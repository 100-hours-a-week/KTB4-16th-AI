from pathlib import Path

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


async def test_tag_returns_empty_when_nothing_clears_threshold():
    tagger = _tagger(
        image_vector=[0.0, 0.0, 1.0],
        tag_vectors={"sunset": [1.0, 0.0, 0.0], "night street": [0.0, 1.0, 0.0]},
    )

    assert await tagger.tag("https://img/1.jpg") == []
