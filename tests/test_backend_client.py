"""AI → 백엔드 콜백 본문: batchRequestId를 값 그대로 돌려주는지."""

import json

import httpx
import pytest

from app.clients.backend_client import BackendCallbackError, BackendClient, FakeBackendClient


def _client(handler) -> BackendClient:
    client = BackendClient(base_url="http://backend", internal_token="t", timeout=5.0)
    client._http = httpx.AsyncClient(
        base_url="http://backend", transport=httpx.MockTransport(handler)
    )
    return client


async def _notify(client: BackendClient, **extra) -> None:
    await client.notify_report_ready(
        job_id="report_batch_2026-9",
        year=2026,
        month=9,
        generated_at="2026-10-01T00:00:00+00:00",
        results=[],
        **extra,
    )


async def test_callback_sends_batch_request_id_as_is():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["token"] = request.headers["X-Internal-Token"]
        return httpx.Response(200)

    await _notify(_client(handler), batch_request_id="5d1c2e7a-0b3f-4f55-9a41-aaaa")

    assert seen["body"]["batchRequestId"] == "5d1c2e7a-0b3f-4f55-9a41-aaaa"
    assert seen["body"]["jobId"] == "report_batch_2026-9"
    assert seen["token"] == "t"


async def test_callback_omits_field_when_request_had_no_batch_request_id():
    """구버전 요청은 기존 본문 그대로 — null을 보내면 백엔드가 '모르는 ID'로 볼 수 있다."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200)

    await _notify(_client(handler))

    assert "batchRequestId" not in seen["body"]


@pytest.mark.parametrize("status", [400, 404, 500, 503])
async def test_callback_error_status_raises_so_job_retries(status):
    """2xx가 아니면 예외 → worker_main이 job을 재시도하고, 다 쓰면 포기한다."""
    client = _client(lambda request: httpx.Response(status, text="no"))

    with pytest.raises(BackendCallbackError):
        await _notify(client, batch_request_id="abc")


async def test_fake_client_records_batch_request_id():
    fake = FakeBackendClient()

    await fake.notify_report_ready(
        job_id="j", year=2026, month=9, generated_at="x", results=[], batch_request_id="abc"
    )

    assert fake.calls[0]["batch_request_id"] == "abc"
