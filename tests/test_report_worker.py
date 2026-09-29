import pytest

from app.workers.report_worker import PhotoSceneStat, _describe_weather, _photo_scene_stats


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
