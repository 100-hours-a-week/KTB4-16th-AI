"""/api/photo-recommend 엔드포인트 배선(DI) 통합 테스트.

fake 모드의 FakeClipClient·FakeLLMClient·FakeSpotifySearchClient를 그대로 태워
get_clip_tagger·get_song_curator·get_reranker·get_spotify_search_client 등 실제
의존성 그래프가 안 깨졌는지 확인한다. Reranker가 태그 겹침만 계산하는 순수
함수로 바뀌면서(2026-09-26) DB 세션이 필요 없어져, 더 이상 오버라이드가
필요 없다 — 실제 의존성 그대로 통합 테스트가 가능해졌다.
"""

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import AUTH


def _client() -> TestClient:
    return TestClient(create_app())


def test_photo_recommend_returns_valid_shape():
    res = _client().post(
        "/api/photo-recommend",
        json={"imageUrl": "https://cdn.muro.app/uploads/img_882.jpg"},
        headers=AUTH,
    )

    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["moodTags"], list)
    assert isinstance(body["rewrittenQuery"], str)
    assert isinstance(body["tracks"], list)
    assert len(body["tracks"]) <= 3
    assert isinstance(body["degraded"], bool)
    for track in body["tracks"]:
        assert {"title", "artistName", "externalTrackId"} <= track.keys()


def test_photo_recommend_needs_token():
    res = _client().post(
        "/api/photo-recommend", json={"imageUrl": "https://cdn.muro.app/uploads/img_882.jpg"}
    )
    assert res.status_code == 401


def test_photo_recommend_missing_image_url_is_400():
    res = _client().post("/api/photo-recommend", json={}, headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "imageUrl"


def test_validation_failure_is_logged_without_values(caplog):
    """400만 보이면 원인을 못 찾는다 — 필드·이유·Content-Type은 남기고 보낸 값은 남기지 않는다."""
    secret_url = "https://storage.googleapis.com/b/a.jpg?X-Goog-Signature=secret"
    with caplog.at_level("WARNING", logger="muro.api"):
        res = _client().post(
            "/api/photo-recommend",
            content=f'{{"imageUrl": "{secret_url}"}}',
            headers={**AUTH, "Content-Type": "text/plain"},
        )

    assert res.status_code == 400
    assert "요청 검증 실패 POST /api/photo-recommend (Content-Type: text/plain)" in caplog.text
    assert "secret" not in caplog.text
