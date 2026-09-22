from pathlib import Path

import pytest

from app.clients.clip_client import FakeClipClient
from app.components.category_vectors import CategoryVectors, load_tag_labels

FIXTURE_YAML = Path(__file__).parent / "fixtures" / "mood_tags_sample.yaml"


def test_load_tag_labels_reads_clip_and_ko_pairs():
    labels = load_tag_labels(FIXTURE_YAML)
    assert labels == [("sunset", "노을"), ("night street", "밤거리")]


def test_load_tag_labels_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("tags: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tag_labels(empty)


async def test_load_populates_tags_with_vectors():
    cv = CategoryVectors(FakeClipClient(dim=8), path=FIXTURE_YAML)
    assert not cv.loaded

    await cv.load()

    assert cv.loaded
    assert [t.ko for t in cv.tags] == ["노을", "밤거리"]
    assert all(len(t.vector) == 8 for t in cv.tags)


async def test_load_is_deterministic_for_same_client():
    cv1 = CategoryVectors(FakeClipClient(dim=8), path=FIXTURE_YAML)
    cv2 = CategoryVectors(FakeClipClient(dim=8), path=FIXTURE_YAML)
    await cv1.load()
    await cv2.load()
    assert cv1.tags[0].vector == cv2.tags[0].vector


def test_default_path_points_at_app_mood_tags_yaml():
    from app.components.category_vectors import DEFAULT_TAGS_PATH

    assert DEFAULT_TAGS_PATH.name == "mood_tags.yaml"
    assert DEFAULT_TAGS_PATH.parent.name == "app"
    assert DEFAULT_TAGS_PATH.exists()
