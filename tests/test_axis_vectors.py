import math
from pathlib import Path

import pytest

from app.clients.clip_client import FakeClipClient
from app.components.axis_vectors import (
    DEFAULT_AXES_PATH,
    AxisTagVector,
    AxisVectors,
    LoadedAxis,
    load_axis_specs,
    pick_tag,
)

FIXTURE_YAML = Path(__file__).parent / "fixtures" / "axis_tags_sample.yaml"


class TextMapClient:
    """문장마다 벡터를 고정해 주고, encode_text가 몇 번 어떤 문장으로 불렸는지 기록한다."""

    model = "text-map-clip"
    dim = 2

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    async def encode_image(self, image_url: str) -> list[float]:
        raise AssertionError("태그 벡터를 만들 때 이미지 인코딩을 부르면 안 된다")

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vectors[t] for t in texts]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "axes.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_axis_specs_reads_axes_in_order_with_when_and_sentences():
    specs = load_axis_specs(FIXTURE_YAML)

    assert [s.key for s in specs] == ["kind", "subject", "expression"]
    assert [s.when for s in specs] == [None, "진짜 사진", "그림·짤"]
    assert specs[0].tags[0].ko == "진짜 사진"
    assert specs[0].tags[0].sentences == ("real photo", "camera photo")
    assert math.isclose(specs[1].margin, 0.1)


@pytest.mark.parametrize(
    "body",
    [
        "axes: []\n",
        # 태그에 문장이 없음
        "axes:\n  - {key: kind, ko: 종류, tags: [{ko: a, sentences: []}]}\n",
        # 축에 태그가 없음
        "axes:\n  - {key: kind, ko: 종류, tags: []}\n",
        # 1단계 축(when 없음)이 둘
        "axes:\n"
        "  - {key: a, ko: A, tags: [{ko: x, sentences: [s]}]}\n"
        "  - {key: b, ko: B, tags: [{ko: y, sentences: [s]}]}\n",
        # when이 1단계 태그에 없음
        "axes:\n"
        "  - {key: a, ko: A, tags: [{ko: x, sentences: [s]}]}\n"
        "  - {key: b, ko: B, when: 없는태그, tags: [{ko: y, sentences: [s]}]}\n",
        # key 중복
        "axes:\n"
        "  - {key: a, ko: A, tags: [{ko: x, sentences: [s]}]}\n"
        "  - {key: a, ko: B, when: x, tags: [{ko: y, sentences: [s]}]}\n",
        # margin이 음수
        "axes:\n  - {key: a, ko: A, margin: -0.1, tags: [{ko: x, sentences: [s]}]}\n",
    ],
)
def test_load_axis_specs_rejects_broken_files(tmp_path, body):
    with pytest.raises(ValueError):
        load_axis_specs(_write(tmp_path, body))


async def test_load_encodes_all_sentences_in_one_call_and_averages_unit_vectors(tmp_path):
    path = _write(
        tmp_path,
        "axes:\n  - key: kind\n    ko: 종류\n    tags:\n"
        "      - {ko: 태그, sentences: [long, short]}\n",
    )
    client = TextMapClient({"long": [3.0, 0.0], "short": [0.0, 4.0]})
    axis_vectors = AxisVectors(client, path=path)

    await axis_vectors.load()

    assert client.calls == [["long", "short"]]  # 문장 전부를 한 번에 요청
    # 길이를 1로 맞춘 (1,0)과 (0,1)의 평균이라 (0.707, 0.707). 길이를 안 맞췄다면
    # (3,0)과 (0,4)의 평균 방향인 (0.6, 0.8)쪽으로 치우친다
    vector = axis_vectors.root.tags[0].vector
    assert all(math.isclose(x, 0.70710678, rel_tol=1e-6) for x in vector)


async def test_load_keeps_yaml_order_and_finds_children_by_root_tag():
    axis_vectors = AxisVectors(FakeClipClient(dim=8), path=FIXTURE_YAML)
    assert not axis_vectors.loaded

    await axis_vectors.load()

    assert axis_vectors.loaded
    assert axis_vectors.root.key == "kind"
    assert [a.key for a in axis_vectors.children_of("진짜 사진")] == ["subject"]
    assert [a.key for a in axis_vectors.children_of("그림·짤")] == ["expression"]
    assert axis_vectors.children_of("없는 태그") == []
    assert axis_vectors.get("subject").ko == "대상"
    with pytest.raises(ValueError):
        axis_vectors.get("weather")


async def test_default_axes_file_matches_the_team_plan():
    """팀 계획의 축·태그 목록이 바뀌면 이 시험을 같이 고친다. 날씨 축은 없어야 한다."""
    axis_vectors = AxisVectors(FakeClipClient(dim=8), path=DEFAULT_AXES_PATH)

    await axis_vectors.load()

    by_key = {a.key: a for a in axis_vectors.axes}
    assert list(by_key) == ["kind", "subject", "vibe", "expression"]
    assert [t.ko for t in by_key["kind"].tags] == ["진짜 사진", "그림·짤"]
    assert [t.ko for t in by_key["subject"].tags] == [
        "셀카·1명",
        "커플·두 사람",
        "여러 사람",
        "음식·음료",
        "반려동물",
        "풍경·사물",
    ]
    assert [t.ko for t in by_key["vibe"].tags] == [
        "밝은 햇살",
        "노을빛",
        "푸르고 밝은",
        "흐린 회색빛",
        "밤의 따뜻한 조명",
        "어둡고 차가운 밤",
    ]
    assert [t.ko for t in by_key["expression"].tags] == [
        "웃음·신남",
        "슬픔·눈물",
        "지침·무기력",
        "화남",
        "설렘·사랑",
    ]
    assert by_key["subject"].when == by_key["vibe"].when == "진짜 사진"
    assert by_key["expression"].when == "그림·짤"
    assert all(len(t.vector) == 8 for a in axis_vectors.axes for t in a.tags)


def test_default_axes_file_has_two_or_three_sentences_per_tag():
    for spec in load_axis_specs(DEFAULT_AXES_PATH):
        for tag in spec.tags:
            assert 2 <= len(tag.sentences) <= 3, f"{spec.key}/{tag.ko}"


def _axis(margin: float) -> LoadedAxis:
    return LoadedAxis(
        key="k",
        ko="축",
        margin=margin,
        when=None,
        tags=(
            AxisTagVector(ko="가", vector=[1.0, 0.0]),
            AxisTagVector(ko="나", vector=[0.0, 1.0]),
        ),
    )


def test_pick_tag_returns_closest_tag_when_gap_is_wide_enough():
    assert pick_tag([1.0, 0.2], _axis(margin=0.1)) == "가"


def test_pick_tag_is_uncertain_when_first_and_second_are_too_close():
    # 코사인 유사도 0.707 대 0.707 — 차이가 0이라 margin 0.1 미만
    assert pick_tag([1.0, 1.0], _axis(margin=0.1)) is None


def test_pick_tag_with_zero_margin_never_gives_up_unless_exact_tie_matters():
    assert pick_tag([1.0, 0.9], _axis(margin=0.0)) == "가"
