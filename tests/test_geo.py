import math

import pytest

from app.components.geo import haversine_km, proximity_score


def test_same_point_is_zero_distance():
    assert haversine_km(37.5, 127.0, 37.5, 127.0) == 0.0


def test_one_degree_of_latitude_is_about_111_km():
    # 위도 1도는 어디서나 약 111.2km (지구 반지름 6371km × π/180)
    assert math.isclose(haversine_km(37.0, 127.0, 38.0, 127.0), 111.19, abs_tol=0.1)


def test_longitude_degree_is_shorter_at_seoul_latitude():
    # 서울 위도(약 37.5도)에서 경도 1도는 약 88.2km (위도 1도의 cos(37.5°) 배)
    assert math.isclose(haversine_km(37.5, 126.0, 37.5, 127.0), 88.2, abs_tol=0.3)


def test_distance_is_symmetric():
    a = haversine_km(37.5436, 127.0447, 37.5575, 126.9248)
    b = haversine_km(37.5575, 126.9248, 37.5436, 127.0447)
    assert math.isclose(a, b)


def test_seoul_forest_to_hongdae_is_about_ten_km():
    # 서울숲 근처 ↔ 홍대입구역: 지도상 약 10~11km
    assert 10.0 < haversine_km(37.5436, 127.0447, 37.5575, 126.9248) < 11.5


def test_proximity_is_one_at_center_and_zero_at_or_beyond_radius():
    assert proximity_score(0.0, 1.5) == 1.0
    assert proximity_score(1.5, 1.5) == 0.0
    assert proximity_score(9.0, 1.5) == 0.0


def test_proximity_falls_linearly_inside_radius():
    assert math.isclose(proximity_score(0.75, 1.5), 0.5)


@pytest.mark.parametrize("radius", [0, -1.0])
def test_non_positive_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        proximity_score(1.0, radius)
