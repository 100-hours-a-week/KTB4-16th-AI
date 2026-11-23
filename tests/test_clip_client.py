import json

import httpx
import pytest

from app.clients.clip_client import ReplicateClipClient, RunPodClipClient
from app.config import Settings
from app.dependencies import _build_clip_client
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


# --- RunPod CLIP 서버 ---


def make_runpod_client(handler, max_retries: int = 1) -> RunPodClipClient:
    return RunPodClipClient(
        base_url="https://pod-8001.proxy.runpod.net/",
        api_key="clip-key",
        model="openai/clip",
        dim=DIM,
        timeout=5,
        max_retries=max_retries,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(RunPodClipClient, "RETRY_WAIT_SECONDS", 0)


async def test_runpod_encode_image_sends_url_with_key_and_user_agent():
    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://pod-8001.proxy.runpod.net/encode_image"
        assert req.headers["Authorization"] == "Bearer clip-key"
        assert req.headers["User-Agent"] == RunPodClipClient.USER_AGENT
        assert json.loads(req.content) == {"image": "https://img/1.jpg"}
        return httpx.Response(200, json={"embedding": [0.1] * DIM})

    client = make_runpod_client(handler)
    assert await client.encode_image("https://img/1.jpg") == [0.1] * DIM
    # 저장·태그 버전에 쓰는 이름은 Replicate 때와 같다
    assert client.model == "openai/clip"


async def test_runpod_encode_text_sends_texts_in_batches(monkeypatch):
    monkeypatch.setattr(RunPodClipClient, "MAX_TEXTS_PER_REQUEST", 2)
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/encode_text"
        texts = json.loads(req.content)["texts"]
        sent.append(texts)
        return httpx.Response(200, json={"embeddings": [[float(len(t))] * DIM for t in texts]})

    vectors = await make_runpod_client(handler).encode_text(["a", "bb", "ccc"])
    assert sent == [["a", "bb"], ["ccc"]]
    assert vectors == [[1.0] * DIM, [2.0] * DIM, [3.0] * DIM]


async def test_runpod_encode_text_empty_makes_no_call():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("빈 목록이면 서버를 부르지 않는다")

    assert await make_runpod_client(handler).encode_text([]) == []


async def test_runpod_wrong_dimension_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1] * (DIM + 1)})

    with pytest.raises(ClipError, match="차원"):
        await make_runpod_client(handler).encode_image("https://img/1.jpg")


async def test_runpod_embedding_count_mismatch_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * DIM]})

    with pytest.raises(ClipError, match="개수"):
        await make_runpod_client(handler).encode_text(["a", "b"])


async def test_runpod_retries_while_pod_restarts(no_retry_wait):
    statuses = iter([502, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"embedding": [0.3] * DIM})

    assert await make_runpod_client(handler).encode_image("https://img/1.jpg") == [0.3] * DIM


async def test_runpod_retries_connection_errors_then_gives_up(no_retry_wait):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("pod down")

    with pytest.raises(ClipError, match="연결 실패"):
        await make_runpod_client(handler, max_retries=2).encode_image("https://img/1.jpg")
    assert len(calls) == 3


@pytest.mark.parametrize("status", [400, 401, 413])
async def test_runpod_client_errors_are_not_retried(no_retry_wait, status):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"detail": "x"})

    with pytest.raises(ClipError, match=str(status)):
        await make_runpod_client(handler).encode_image("https://img/1.jpg")
    assert len(calls) == 1


# --- CLIP_PROVIDER로 고르기 ---


def _settings(monkeypatch, **values) -> Settings:
    for name in ("CLIP_PROVIDER", "CLIP_SERVER_URL", "CLIP_API_KEY", "CLIP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, **values)


def test_clip_provider_default_is_replicate(monkeypatch):
    s = _settings(monkeypatch)
    assert s.clip_provider == "replicate"
    client = _build_clip_client(s)
    assert isinstance(client, ReplicateClipClient)
    assert client.model == "openai/clip"


def test_clip_provider_runpod_builds_runpod_client(monkeypatch):
    s = _settings(
        monkeypatch,
        clip_provider="runpod",
        clip_server_url="https://pod-8001.proxy.runpod.net",
        clip_api_key="clip-key",
    )
    client = _build_clip_client(s)
    assert isinstance(client, RunPodClipClient)
    assert client.model == "openai/clip"
    assert client.dim == 768
