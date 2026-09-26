"""/api/context-recommend 엔드포인트 배선(DI) 통합 테스트.

fake 모드 클라이언트로 실제 의존성 그래프를 태운다. 과거 자물쇠 검색만 실 DB가 필요해서
빈 결과를 돌려주는 가짜로 바꾼다.
"""

from fastapi.testclient import TestClient

from app.dependencies import get_record_search
from app.main import create_app
from tests.conftest import AUTH


class EmptyRecordSearch:
    async def similar_by_comment(self, user_id, query_vector, limit):
        return []


def _client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_record_search] = lambda: EmptyRecordSearch()
    return TestClient(app)


PAYLOAD = {
    "requestId": "req_ctx_5521",
    "userId": 1044,
    "place": {"placeId": 10, "name": "동탄역 카페거리", "lat": 37.2, "lng": 127.075},
    "weather": {"condition": "RAIN", "temperature": 18},
    "localTime": "2026-08-31T19:40:00+09:00",
    "limit": 3,
}


def test_context_recommend_returns_valid_shape():
    res = _client().post("/api/context-recommend", json=PAYLOAD, headers=AUTH)

    assert res.status_code == 200
    body = res.json()
    assert body["requestId"] == "req_ctx_5521"
    assert body["recommendationBasis"] in {"PERSONAL", "REGIONAL", "GENERIC"}
    assert isinstance(body["tracks"], list)
    assert len(body["tracks"]) <= 3
    assert isinstance(body["degraded"], bool)
    for track in body["tracks"]:
        assert {"title", "artist", "externalTrackId", "rerankScore"} <= track.keys()


def test_context_recommend_missing_user_id_is_400():
    payload = {k: v for k, v in PAYLOAD.items() if k != "userId"}
    res = _client().post("/api/context-recommend", json=payload, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == "userId"


def test_string_user_id_is_rejected():
    """백엔드 명세상 userId는 숫자(Long). 위키 예시("u_1044")처럼 문자열은 받지 않는다."""
    res = _client().post(
        "/api/context-recommend", json={**PAYLOAD, "userId": "u_1044"}, headers=AUTH
    )

    assert res.status_code == 400
    assert res.json()["field"] == "userId"
