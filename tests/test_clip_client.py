import httpx
import pytest

from app.clients.clip_client import ReplicateClipClient
from app.exceptions import ClipError

DIM = 4


def make_client(handler) -> ReplicateClipClient:
    return ReplicateClipClient(
        api_token="test",
        model="openai/clip",
        dim=DIM,
        timeout=5,
        max_retries=1,
        transport=httpx.MockTransport(handler),
    )


async def test_encode_image_sync_success():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/models/openai/clip/predictions"
        assert req.headers["Authorization"] == "Bearer test"
        assert req.headers["Prefer"].startswith("wait=")
        assert b'"image":"https://img/1.jpg"' in req.content.replace(b" ", b"")
        return httpx.Response(
            201, json={"id": "p1", "status": "succeeded", "output": {"embedding": [0.1] * DIM}}
        )

    assert await make_client(handler).encode_image("https://img/1.jpg") == [0.1] * DIM


async def test_polls_until_succeeded(monkeypatch):
    monkeypatch.setattr(ReplicateClipClient, "POLL_INTERVAL_SECONDS", 0)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.method)
        if req.method == "POST":
            return httpx.Response(201, json={"id": "p1", "status": "processing"})
        return httpx.Response(
            200, json={"id": "p1", "status": "succeeded", "output": {"embedding": [0.2] * DIM}}
        )

    assert await make_client(handler).encode_image("https://img/1.jpg") == [0.2] * DIM
    assert calls == ["POST", "GET"]


async def test_encode_text_sends_each_text():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.content)
        return httpx.Response(
            201, json={"id": "p", "status": "succeeded", "output": {"embedding": [0.3] * DIM}}
        )

    vectors = await make_client(handler).encode_text(["노을", "카페"])
    assert len(vectors) == 2 and len(seen) == 2


async def test_failed_prediction_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": "p", "status": "failed", "error": "bad image"})

    with pytest.raises(ClipError):
        await make_client(handler).encode_image("https://img/x.jpg")


async def test_wrong_dimension_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201, json={"id": "p", "status": "succeeded", "output": {"embedding": [0.1] * 3}}
        )

    with pytest.raises(ClipError):
        await make_client(handler).encode_image("https://img/x.jpg")


async def test_auth_error_is_not_retried():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"detail": "Unauthenticated"})

    with pytest.raises(ClipError):
        await make_client(handler).encode_image("https://img/x.jpg")
    assert len(calls) == 1


async def test_rate_limit_waits_and_retries(monkeypatch):
    monkeypatch.setattr(ReplicateClipClient, "RATE_LIMIT_WAIT_SECONDS", 0)
    statuses = [429, 429, 201]

    def handler(req: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 429:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"detail": "rate"})
        return httpx.Response(
            201, json={"id": "p", "status": "succeeded", "output": {"embedding": [0.5] * DIM}}
        )

    assert await make_client(handler).encode_image("https://img/1.jpg") == [0.5] * DIM
    assert statuses == []


async def test_rate_limit_gives_up_eventually(monkeypatch):
    monkeypatch.setattr(ReplicateClipClient, "RATE_LIMIT_WAIT_SECONDS", 0)
    monkeypatch.setattr(ReplicateClipClient, "RATE_LIMIT_EXTRA_RETRIES", 2)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"detail": "rate"})

    with pytest.raises(ClipError):
        await make_client(handler).encode_image("https://img/1.jpg")
    assert len(calls) == 3
