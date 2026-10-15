import pytest

from app.components.report_summarizer import PhotoAxisStat
from app.db.repositories.report_repository import StoredPhotoTags
from app.exceptions import ClipError
from app.workers import report_worker
from app.workers.report_worker import (
    MIN_AXIS_PHOTOS,
    BatchScope,
    PhotoSceneStat,
    _axis_stats,
    _classify_axes,
    _classify_photos,
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


# --- 저장된 태그를 읽는 경로 ---------------------------------------------------------

VERSION = "tv-now"
SAVED = StoredPhotoTags(
    tags={"장소": "카페", "사진 종류": "진짜 사진", "대상": "여러 사람"}, tag_version=VERSION
)


class RecordingTagger:
    """벡터로 다시 분류할 때만 불린다. 불린 사진을 기록해서 "다시 계산했는가"를 확인한다."""

    def __init__(self, error: Exception | None = None):
        self._error = error
        self.scene_calls: list[list[float]] = []
        self.axes_calls: list[list[float]] = []

    async def tag_from_vector(self, vector, top_k=3, **kwargs):
        self.scene_calls.append(vector)
        return ["노을"]

    async def classify_axes(self, vector):
        if self._error:
            raise self._error
        self.axes_calls.append(vector)
        return {"사진 종류": "그림·짤", "표정·감정": "웃음·신남"}


def _tagger_must_not_be_built(monkeypatch):
    def boom():
        raise AssertionError("저장된 태그만으로 충분하면 태거를 만들 필요도 없다")

    monkeypatch.setattr(report_worker, "get_clip_tagger", boom)


async def test_photos_use_the_saved_scene_tag_without_reclassifying(monkeypatch):
    _tagger_must_not_be_built(monkeypatch)

    result = await _classify_photos({1: [0.1]}, {1: SAVED}, VERSION)

    assert result == {1: ["카페"]}


async def test_photos_without_saved_tags_are_classified_from_the_vector(monkeypatch):
    tagger = RecordingTagger()
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: tagger)
    stored = {
        1: StoredPhotoTags(tags=None, tag_version=None),  # 예전에 저장된 행·분류 실패
        2: StoredPhotoTags(tags=SAVED.tags, tag_version="tv-old"),  # 다른 태그 목록으로 분류됨
    }

    result = await _classify_photos({1: [0.1], 2: [0.2], 3: [0.3]}, stored, VERSION)

    assert result == {1: ["노을"], 2: ["노을"], 3: ["노을"]}  # 3은 저장된 행 자체가 없다
    assert tagger.scene_calls == [[0.1], [0.2], [0.3]]


async def test_saved_empty_tags_mean_no_confident_tag_not_reclassify(monkeypatch):
    """{}는 "분류했지만 붙일 태그가 없음"이다. 다시 계산하지 않고 기타로 센다."""
    _tagger_must_not_be_built(monkeypatch)
    stored = {1: StoredPhotoTags(tags={}, tag_version=VERSION)}

    assert await _classify_photos({1: [0.1]}, stored, VERSION) == {1: []}


async def test_photos_mix_saved_and_recomputed_in_one_run(monkeypatch):
    tagger = RecordingTagger()
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: tagger)

    result = await _classify_photos({1: [0.1], 2: [0.2]}, {1: SAVED}, VERSION)

    assert result == {1: ["카페"], 2: ["노을"]}
    assert tagger.scene_calls == [[0.2]]  # 저장된 1번은 다시 계산하지 않았다


async def test_without_a_current_tag_version_nothing_saved_is_trusted(monkeypatch):
    tagger = RecordingTagger()
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: tagger)

    result = await _classify_photos({1: [0.1]}, {1: SAVED}, None)

    assert result == {1: ["노을"]}


async def test_axes_use_saved_tags_and_leave_out_the_scene_tag(monkeypatch):
    _tagger_must_not_be_built(monkeypatch)

    result = await _classify_axes({1: [0.1]}, {1: SAVED}, VERSION)

    assert result == {1: {"사진 종류": "진짜 사진", "대상": "여러 사람"}}  # "장소"는 다축이 아님


async def test_axes_recompute_only_the_photos_without_usable_saved_tags(monkeypatch):
    tagger = RecordingTagger()
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: tagger)

    result = await _classify_axes({2: [0.2], 1: [0.1]}, {1: SAVED}, VERSION)

    assert list(result) == [2, 1]  # 사진 순서는 그대로
    assert result[1] == {"사진 종류": "진짜 사진", "대상": "여러 사람"}
    assert result[2] == {"사진 종류": "그림·짤", "표정·감정": "웃음·신남"}
    assert tagger.axes_calls == [[0.2]]


async def test_axes_clip_failure_keeps_saved_tags_and_skips_only_the_rest(monkeypatch):
    tagger = RecordingTagger(error=ClipError("태그 문장 인코딩 실패"))
    monkeypatch.setattr(report_worker, "get_clip_tagger", lambda: tagger)

    result = await _classify_axes({1: [0.1], 2: [0.2]}, {1: SAVED}, VERSION)

    assert result == {1: {"사진 종류": "진짜 사진", "대상": "여러 사람"}}
