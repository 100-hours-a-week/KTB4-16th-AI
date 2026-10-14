import pytest

from app.components.report_summarizer import PhotoAxisStat
from app.exceptions import ClipError
from app.workers import report_worker
from app.workers.report_worker import (
    MIN_AXIS_PHOTOS,
    BatchScope,
    PhotoSceneStat,
    _axis_stats,
    _classify_axes,
    _describe_weather,
    _photo_scene_stats,
)


def test_batch_scope_with_id_uses_it_for_job_and_notify_keys():
    scope = BatchScope(year=2026, month=9, batch_request_id="req-A")

    assert scope.job_prefix == "req-A:"
    assert scope.job_key(12) == "req-A:12"
    assert scope.notify_key == "batch:req-A"


def test_batch_scope_without_id_keeps_year_month_keys():
    scope = BatchScope(year=2026, month=9)

    assert scope.job_prefix == "2026-9-"
    assert scope.job_key(12) == "2026-9-12"
    assert scope.notify_key == "2026-9"


def test_one_id_prefix_never_matches_another_ids_jobs():
    """접두사로 배치의 job을 세므로, 'abc'가 'abc-1' 배치의 job을 세면 안 된다."""
    a = BatchScope(year=2026, month=9, batch_request_id="abc")
    b = BatchScope(year=2026, month=9, batch_request_id="abc-1")

    assert not b.job_key(1).startswith(a.job_prefix)
    assert not a.job_key(1).startswith(b.job_prefix)


def test_id_that_looks_like_year_month_does_not_collide_with_legacy_keys():
    legacy = BatchScope(year=2026, month=9)
    tricky = BatchScope(year=2026, month=9, batch_request_id="2026-9")

    assert tricky.notify_key != legacy.notify_key
    assert not tricky.job_key(1).startswith(legacy.job_prefix)
    assert not legacy.job_key(1).startswith(tricky.job_prefix)


def test_scope_from_payload_reads_batch_request_id():
    scope = BatchScope.from_payload(
        {"user_id": 1, "year": 2026, "month": 9, "batch_request_id": "req-A"}
    )

    assert scope == BatchScope(year=2026, month=9, batch_request_id="req-A")


def test_scope_from_payload_of_job_queued_before_this_feature_falls_back_to_year_month():
    """배포 전에 큐에 들어간 job에는 batch_request_id가 없다."""
    scope = BatchScope.from_payload({"user_id": 1, "year": 2026, "month": 9})

    assert scope == BatchScope(year=2026, month=9, batch_request_id=None)
    assert scope.job_prefix == "2026-9-"


@pytest.mark.parametrize(
    ("condition", "korean"),
    [
        ("CLEAR", "맑음"),
        ("CLOUDY", "구름 많음"),
        ("OVERCAST", "흐림"),
        ("RAIN", "비"),
        ("SNOW", "눈"),
        ("RAIN_SNOW", "진눈깨비"),
        ("SHOWER", "소나기"),
    ],
)
def test_every_backend_weather_condition_is_described(condition, korean):
    """백엔드 records.weather_condition ENUM 7종.

    빠지면 영어 코드가 요약 입력에 그대로 들어간다.
    """
    assert _describe_weather(condition) == korean


def test_photo_scene_stats_counts_and_orders_by_frequency_descending():
    photo_tags_by_record = {1: ["카페"], 2: ["카페"], 3: ["노을"]}

    assert _photo_scene_stats(photo_tags_by_record) == [
        PhotoSceneStat(tag="카페", count=2, ratio=67),
        PhotoSceneStat(tag="노을", count=1, ratio=33),
    ]


def test_photo_scene_stats_empty_when_no_records():
    assert _photo_scene_stats({}) == []


def test_axis_stats_counts_only_photos_that_have_a_tag_on_that_axis():
    axes_by_record = {
        1: {"사진 종류": "진짜 사진", "대상": "음식·음료"},
        2: {"사진 종류": "진짜 사진", "대상": "음식·음료"},
        3: {"사진 종류": "진짜 사진", "대상": "여러 사람"},
        4: {"사진 종류": "진짜 사진"},  # 대상이 불확실해서 빠진 사진 — 대상의 분모에 넣지 않는다
        5: {},  # 사진 종류부터 불확실
    }

    stats = _axis_stats(axes_by_record)

    assert stats["사진 종류"] == PhotoAxisStat(count=4, ratios={"진짜 사진": 1.0})
    assert stats["대상"].count == 3
    assert stats["대상"].ratios == {"음식·음료": 2 / 3, "여러 사람": 1 / 3}


def test_axis_stats_drops_axes_with_too_few_photos():
    axes_by_record = {i: {"표정·감정": "웃음·신남"} for i in range(MIN_AXIS_PHOTOS - 1)} | {
        99: {"대상": "반려동물"}
    }

    assert _axis_stats(axes_by_record) == {}


def test_axis_stats_empty_when_no_photos():
    assert _axis_stats({}) == {}


class StubAxisTagger:
    def __init__(self, error: Exception | None = None):
        self._error = error

    async def classify_axes(self, vector):
        if self._error:
            raise self._error
        return {"사진 종류": "진짜 사진"}


async def test_classify_axes_returns_tags_by_record(monkeypatch):
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: StubAxisTagger())

    result = await _classify_axes({1: [0.1], 2: [0.2]})

    assert result == {1: {"사진 종류": "진짜 사진"}, 2: {"사진 종류": "진짜 사진"}}


async def test_classify_axes_skips_instead_of_failing_the_report_when_clip_is_down(monkeypatch):
    """태그 문장을 CLIP으로 인코딩하지 못해도 RECAP 요약은 계속된다 (사진 특징만 빠진다)."""
    monkeypatch.setattr(
        report_worker, "get_clip_tagger", lambda: StubAxisTagger(ClipError("CLIP 장애"))
    )

    assert await _classify_axes({1: [0.1]}) == {}


async def test_classify_axes_does_not_touch_clip_when_there_are_no_photos(monkeypatch):
    def boom():
        raise AssertionError("사진이 없으면 태거를 만들 필요도 없다")

    monkeypatch.setattr(report_worker, "get_clip_tagger", boom)

    assert await _classify_axes({}) == {}
