import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.dependencies as dependencies
from app.clients.clip_client import FakeClipClient
from app.components.axis_vectors import AxisVectors, pick_tag
from app.components.category_vectors import CategoryVectors
from app.components.tag_vector_file import (
    DEFAULT_VECTORS_PATH,
    PrecomputedVectors,
    from_loaded,
    read_vectors_file,
    write_vectors_file,
)
from app.components.tag_version import compute_tag_version
from app.config import get_settings

FIXTURES = Path(__file__).parent / "fixtures"
SCENE_YAML = FIXTURES / "mood_tags_sample.yaml"  # 라벨: sunset, night street
AXIS_YAML = FIXTURES / "axis_tags_sample.yaml"  # 축: kind, subject, expression
DIM = 4


def _vectors(dim: int = DIM) -> PrecomputedVectors:
    def v(i: int) -> list[float]:
        return [1.0 if k == i else 0.0 for k in range(dim)]

    return PrecomputedVectors(
        version="v1",
        model="test-clip",
        scene={"sunset": v(0), "night street": v(1)},
        axes={
            "kind": {"진짜 사진": v(0), "그림·짤": v(1)},
            "subject": {"음식": v(2), "풍경": v(3)},
            "expression": {"웃음": v(0), "슬픔": v(1)},
        },
    )


class NoClip:
    """CLIP을 부르면 시험이 실패한다 — 미리 계산한 벡터만으로 준비돼야 하는 경로용."""

    model = "no-clip"
    dim = DIM

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("미리 계산한 벡터가 있으면 CLIP을 부르면 안 된다")


class CountingClip(FakeClipClient):
    def __init__(self, dim: int = DIM):
        super().__init__(dim=dim)
        self.calls = 0

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        await asyncio.sleep(0.01)  # 동시에 부른 다른 요청이 끼어들 틈을 준다
        return await super().encode_text(texts)


# --- 파일 읽기·쓰기 ---------------------------------------------------------------


def test_write_then_read_gives_the_same_vectors_rounded_to_six_decimals(tmp_path):
    path = tmp_path / "v.json"
    vectors = PrecomputedVectors(
        version="v1",
        model="m",
        scene={"sunset": [0.123456789, -0.5]},
        axes={"kind": {"진짜 사진": [1.0, 0.0000004]}},
    )

    write_vectors_file(path, vectors)
    read = read_vectors_file(path)

    assert read is not None
    assert (read.version, read.model) == ("v1", "m")
    assert read.scene == {"sunset": [0.123457, -0.5]}
    assert read.axes == {"kind": {"진짜 사진": [1.0, 0.0]}}


@pytest.mark.parametrize(
    "body",
    [
        "이건 JSON이 아님",
        "{}",
        '{"version": "v1", "model": "m", "scene": {}}',  # axes 없음
        '{"version": "v1", "model": "m", "scene": [], "axes": {}}',  # scene이 목록
        '{"version": "v1", "model": "m", "scene": {"a": ["x"]}, "axes": {}}',  # 숫자가 아님
    ],
)
def test_broken_file_is_treated_as_missing(tmp_path, body):
    path = tmp_path / "v.json"
    path.write_text(body, encoding="utf-8")

    assert read_vectors_file(path) is None


def test_missing_file_is_treated_as_missing(tmp_path):
    assert read_vectors_file(tmp_path / "없는파일.json") is None


# --- 미리 계산한 벡터로 준비하기 -----------------------------------------------------


async def test_scene_tags_are_ready_from_the_file_without_calling_clip():
    categories = CategoryVectors(NoClip(), path=SCENE_YAML, precomputed=_vectors())

    await categories.load()

    assert categories.loaded
    assert [(t.clip_label, t.ko) for t in categories.tags] == [
        ("sunset", "노을"),
        ("night street", "밤거리"),
    ]
    assert categories.tags[0].vector == [1.0, 0.0, 0.0, 0.0]


async def test_axes_are_ready_from_the_file_without_calling_clip():
    axes = AxisVectors(NoClip(), path=AXIS_YAML, precomputed=_vectors())

    await axes.load()

    assert axes.loaded
    assert [a.key for a in axes.axes] == ["kind", "subject", "expression"]
    # 파일에서 읽은 벡터로 분류가 된다: 음식 방향 사진
    assert pick_tag([0.0, 0.0, 1.0, 0.0], axes.get("subject")) == "음식"


async def test_scene_tags_fall_back_to_clip_when_the_file_misses_a_label():
    clip = CountingClip()
    broken = PrecomputedVectors("v1", "m", scene={"sunset": [1.0] * DIM}, axes={})

    categories = CategoryVectors(clip, path=SCENE_YAML, precomputed=broken)
    await categories.load()

    assert clip.calls == 1
    assert len(categories.tags) == 2


async def test_scene_tags_fall_back_to_clip_when_the_dimension_differs():
    clip = CountingClip(dim=DIM)
    wrong_dim = _vectors(dim=DIM + 2)  # 파일은 6차원, 지금 CLIP은 4차원

    categories = CategoryVectors(clip, path=SCENE_YAML, precomputed=wrong_dim)
    await categories.load()

    assert clip.calls == 1
    assert all(len(t.vector) == DIM for t in categories.tags)


async def test_axes_fall_back_to_clip_when_a_tag_is_missing_in_the_file():
    clip = CountingClip()
    vectors = _vectors()
    incomplete = PrecomputedVectors(
        "v1", "m", scene=vectors.scene, axes={**vectors.axes, "subject": {"음식": [0.0] * DIM}}
    )

    axes = AxisVectors(clip, path=AXIS_YAML, precomputed=incomplete)
    await axes.load()

    assert clip.calls == 1
    assert len(axes.get("subject").tags) == 2


async def test_axes_fall_back_to_clip_when_the_dimension_differs():
    clip = CountingClip(dim=DIM)

    axes = AxisVectors(clip, path=AXIS_YAML, precomputed=_vectors(dim=DIM + 2))
    await axes.load()

    assert clip.calls == 1


async def test_without_a_file_both_still_work_through_clip_as_before():
    clip = CountingClip()
    categories = CategoryVectors(clip, path=SCENE_YAML)
    axes = AxisVectors(clip, path=AXIS_YAML)

    await categories.load()
    await axes.load()

    assert categories.loaded and axes.loaded
    assert clip.calls == 2


# --- 동시에 처음 불릴 때 한 번만 계산 --------------------------------------------------


async def test_concurrent_first_loads_encode_scene_tags_only_once():
    clip = CountingClip()
    categories = CategoryVectors(clip, path=SCENE_YAML)

    await asyncio.gather(*(categories.load() for _ in range(5)))

    assert clip.calls == 1


async def test_concurrent_first_loads_encode_axis_sentences_only_once():
    clip = CountingClip()
    axes = AxisVectors(clip, path=AXIS_YAML)

    await asyncio.gather(*(axes.load() for _ in range(5)))

    assert clip.calls == 1


# --- 계산 결과를 파일로 옮겨도 같은 벡터 -----------------------------------------------


async def test_file_made_from_live_vectors_gives_the_same_tags_back(tmp_path):
    clip = FakeClipClient(dim=8)
    live_categories = CategoryVectors(clip, path=SCENE_YAML)
    live_axes = AxisVectors(clip, path=AXIS_YAML)
    await live_categories.load()
    await live_axes.load()
    path = tmp_path / "v.json"
    write_vectors_file(path, from_loaded("v1", clip.model, live_categories, live_axes))

    restored = read_vectors_file(path)
    # 파일의 벡터는 8차원이라, 차원을 맞춘 "CLIP을 부르면 실패하는" 가짜로 확인한다
    no_clip_8 = SimpleNamespace(dim=8, model="x", encode_text=NoClip().encode_text)
    categories = CategoryVectors(no_clip_8, path=SCENE_YAML, precomputed=restored)
    axes = AxisVectors(no_clip_8, path=AXIS_YAML, precomputed=restored)
    await categories.load()
    await axes.load()

    def rounded(vector):
        return [round(x, 6) for x in vector]

    assert [rounded(t.vector) for t in categories.tags] == [
        rounded(t.vector) for t in live_categories.tags
    ]
    for live, restored_axis in zip(live_axes.axes, axes.axes, strict=True):
        assert [rounded(t.vector) for t in restored_axis.tags] == [
            rounded(t.vector) for t in live.tags
        ]


# --- 서버가 파일을 쓸지 정하는 규칙 ----------------------------------------------------


@pytest.fixture
def clean_cache():
    dependencies.get_precomputed_vectors.cache_clear()
    yield
    dependencies.get_precomputed_vectors.cache_clear()


def _stub_runtime(monkeypatch, file_vectors, *, current_version: str):
    monkeypatch.setattr(dependencies, "read_vectors_file", lambda: file_vectors)
    monkeypatch.setattr(dependencies, "compute_tag_version", lambda model: current_version)
    monkeypatch.setattr(
        dependencies, "get_clients", lambda: SimpleNamespace(clip=SimpleNamespace(model="m"))
    )


def test_server_uses_the_file_when_its_version_matches(monkeypatch, clean_cache):
    vectors = _vectors()
    _stub_runtime(monkeypatch, vectors, current_version="v1")

    assert dependencies.get_precomputed_vectors() is vectors


def test_server_ignores_the_file_when_its_version_differs(monkeypatch, clean_cache, caplog):
    """태그 정의나 CLIP 모델이 바뀌었는데 파일을 안 만든 경우 — 오래된 벡터를 쓰면 안 된다."""
    _stub_runtime(monkeypatch, _vectors(), current_version="v2")

    with caplog.at_level("WARNING", logger="muro.dependencies"):
        assert dependencies.get_precomputed_vectors() is None

    assert "build_tag_vectors.py" in caplog.text  # 무엇을 해야 하는지 로그에 남긴다


def test_server_ignores_a_missing_file(monkeypatch, clean_cache):
    _stub_runtime(monkeypatch, None, current_version="v1")

    assert dependencies.get_precomputed_vectors() is None


# --- 저장소에 들어 있는 실제 파일 -----------------------------------------------------


def test_committed_vector_file_matches_the_current_tag_definitions():
    """yaml의 태그나 문장을 바꾸고 scripts/build_tag_vectors.py를 다시 실행하지 않으면 실패한다.

    그 상태로 배포하면 서버가 파일을 쓰지 않아서 첫 사진 추천이 다시 몇 분씩 걸린다.
    """
    settings = get_settings()
    vectors = read_vectors_file(DEFAULT_VECTORS_PATH)

    assert vectors is not None, "app/tag_vectors.json이 없어요. scripts/build_tag_vectors.py 실행"
    assert vectors.version == compute_tag_version(settings.clip_model), (
        "태그 정의나 CLIP 모델이 바뀌었어요. scripts/build_tag_vectors.py를 다시 실행하세요."
    )
    assert all(len(v) == settings.clip_embedding_dim for v in vectors.scene.values())
    assert all(
        len(v) == settings.clip_embedding_dim
        for tags in vectors.axes.values()
        for v in tags.values()
    )
