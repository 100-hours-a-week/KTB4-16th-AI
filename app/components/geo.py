"""지도 위 두 점 사이 거리 — 기능5 기억 검색의 장소 조건 점수 (순수 함수)."""

import math

EARTH_RADIUS_KM = 6371.0088  # 지구 평균 반지름


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 위경도 사이의 지표면 거리(km). 수백 m~수십 km 규모에서 쓰기에 충분히 정확하다."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def proximity_score(distance_km: float, radius_km: float) -> float:
    """중심에 가까울수록 1, 반경 밖이면 0 (`max(0, 1 - 거리 ÷ 반경)`)."""
    if radius_km <= 0:
        raise ValueError("반경은 0보다 커야 해요.")
    return max(0.0, 1.0 - distance_km / radius_km)
