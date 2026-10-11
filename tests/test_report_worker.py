import pytest

from app.workers.report_worker import (
    BatchScope,
    PhotoSceneStat,
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
