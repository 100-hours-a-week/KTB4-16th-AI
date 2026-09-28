import pytest

from app.workers.report_worker import _describe_weather, _scene_list


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


def test_scene_list_orders_by_frequency_descending():
    distribution = {"카페": 0.25, "노을": 0.5, "산": 0.25}

    assert _scene_list(distribution) == ["노을", "카페", "산"]


def test_scene_list_empty_when_no_distribution():
    assert _scene_list({}) == []
