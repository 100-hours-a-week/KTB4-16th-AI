import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.moderation_proxy import get_moderation_client
from app.main import create_app
from tests.conftest import AUTH

BODY = {"requestId": "req_1", "roomId": "room_1", "userId": "u_1", "text": "안녕"}


def _client(handler) -> tuple[TestClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    app = create_app()
    app.dependency_overrides[get_moderation_client] = lambda: httpx.AsyncClient(
        base_url="http://moderation.local", transport=httpx.MockTransport(record)
    )
    return TestClient(app), seen


def test_check_is_forwarded_as_is_with_the_internal_token():
    reply = {"requestId": "req_1", "isToxic": False, "confidence": 0.1, "category": None}
    client, seen = _client(lambda r: httpx.Response(200, json={**reply, "modelVersion": "v1"}))

    res = client.post("/api/moderation/check", json=BODY, headers=AUTH)

    assert res.status_code == 200
    assert res.json()["modelVersion"] == "v1" and res.json()["isToxic"] is False
    assert len(seen) == 1
    assert seen[0].url.path == "/api/moderation/check"
    assert seen[0].headers["x-internal-token"]  # 모더레이션 서비스도 토큰을 확인한다
    assert b'"requestId"' in seen[0].content and "안녕".encode() in seen[0].content  # 본문은 그대로


def test_errors_from_the_moderation_service_pass_through():
    err = {"code": "MODEL_UNAVAILABLE", "message": "모델을 사용할 수 없어요.", "field": None}
    client, _ = _client(lambda r: httpx.Response(503, json=err))

    res = client.post("/api/moderation/check", json=BODY, headers=AUTH)

    assert res.status_code == 503 and res.json()["code"] == "MODEL_UNAVAILABLE"


def test_connection_failure_becomes_503_so_the_backend_blocks_the_message():
    def boom(request):
        raise httpx.ConnectError("refused")

    client, _ = _client(boom)

    res = client.post("/api/moderation/check", json=BODY, headers=AUTH)

    assert res.status_code == 503 and res.json()["code"] == "UPSTREAM_UNAVAILABLE"


def test_timeout_also_becomes_503():
    def slow(request):
        raise httpx.ReadTimeout("too slow")

    client, _ = _client(slow)

    assert client.post("/api/moderation/check", json=BODY, headers=AUTH).status_code == 503


def test_token_is_required_on_the_gateway_too():
    client, seen = _client(lambda r: httpx.Response(200, json={}))

    res = client.post("/api/moderation/check", json=BODY)

    assert res.status_code == 401 and seen == []


def test_health_is_forwarded_without_a_token():
    client, seen = _client(lambda r: httpx.Response(200, json={"status": "ok"}))

    res = client.get("/api/moderation/health")

    assert res.status_code == 200 and res.json() == {"status": "ok"}
    assert "x-internal-token" not in seen[0].headers


@pytest.fixture(autouse=True)
def _clear_cached_client():
    get_moderation_client.cache_clear()
    yield
    get_moderation_client.cache_clear()


def test_empty_body_is_logged_at_the_gateway_and_still_forwarded(caplog):
    import logging

    client, seen = _client(lambda r: httpx.Response(400, json={"code": "INVALID_INPUT"}))

    with caplog.at_level(logging.WARNING, logger="muro.moderation_proxy"):
        res = client.post(
            "/api/moderation/check",
            content=b"",
            headers={**AUTH, "Content-Type": "application/json"},
        )

    assert res.status_code == 400  # 안쪽의 판단(본문 필요)을 그대로 전달
    assert any("본문이 비어 있음" in r.message for r in caplog.records)
    assert len(seen) == 1 and seen[0].content == b""
