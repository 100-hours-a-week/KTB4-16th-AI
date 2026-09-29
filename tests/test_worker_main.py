"""worker_main._execute — job 결과 확정 후 후처리(ON_FINISHED) 호출 규칙."""

from contextlib import asynccontextmanager
from typing import Any

import pytest

from app import worker_main
from app.db.repositories.job_repository import ClaimedJob


class FakeJobRepository:
    """상태 확정 순서를 기록한다. mark_failed는 실제처럼 재시도 소진 여부를 돌려준다."""

    events: list[str] = []

    def __init__(self, _session: Any) -> None:
        pass

    async def mark_done(self, job_id: int) -> None:
        self.events.append(f"done:{job_id}")

    async def mark_failed(self, job: ClaimedJob, error: str) -> bool:
        gave_up = job.attempts >= job.max_attempts
        self.events.append(f"{'failed' if gave_up else 'retry'}:{job.id}")
        return gave_up


@pytest.fixture
def events(monkeypatch) -> list[str]:
    FakeJobRepository.events = []

    @asynccontextmanager
    async def fake_session():
        yield None

    monkeypatch.setattr(worker_main, "JobRepository", FakeJobRepository)
    monkeypatch.setattr(worker_main, "get_sessionmaker", lambda: fake_session)
    return FakeJobRepository.events


def _job(*, attempts: int = 1, max_attempts: int = 3) -> ClaimedJob:
    return ClaimedJob(
        id=7,
        job_type="t",
        payload={"user_id": 34},
        attempts=attempts,
        max_attempts=max_attempts,
    )


def _register(monkeypatch, events: list[str], *, fail: bool) -> None:
    async def handler(payload: dict[str, Any]) -> None:
        if fail:
            raise RuntimeError("boom")

    async def on_finished(payload: dict[str, Any]) -> None:
        events.append(f"finished:{payload['user_id']}")

    monkeypatch.setattr(worker_main, "HANDLERS", {"t": handler})
    monkeypatch.setattr(worker_main, "ON_FINISHED", {"t": on_finished})


async def test_success_runs_hook_after_marking_done(monkeypatch, events):
    _register(monkeypatch, events, fail=False)
    await worker_main._execute(_job())
    assert events == ["done:7", "finished:34"]


async def test_final_failure_also_runs_hook(monkeypatch, events):
    """배치의 마지막 job이 실패로 끝나도 완료 알림 체크가 돌아야 한다(2026-09 RECAP 콜백 누락)."""
    _register(monkeypatch, events, fail=True)
    await worker_main._execute(_job(attempts=3, max_attempts=3))
    assert events == ["failed:7", "finished:34"]


async def test_failure_with_retries_left_does_not_run_hook(monkeypatch, events):
    _register(monkeypatch, events, fail=True)
    await worker_main._execute(_job(attempts=1, max_attempts=3))
    assert events == ["retry:7"]


async def test_hook_error_does_not_crash_worker(monkeypatch, events):
    async def handler(payload: dict[str, Any]) -> None:
        pass

    async def broken_hook(payload: dict[str, Any]) -> None:
        raise RuntimeError("backend down")

    monkeypatch.setattr(worker_main, "HANDLERS", {"t": handler})
    monkeypatch.setattr(worker_main, "ON_FINISHED", {"t": broken_hook})
    await worker_main._execute(_job())
    assert events == ["done:7"]


async def test_job_type_without_hook_is_fine(monkeypatch, events):
    async def handler(payload: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr(worker_main, "HANDLERS", {"t": handler})
    monkeypatch.setattr(worker_main, "ON_FINISHED", {})
    await worker_main._execute(_job())
    assert events == ["done:7"]
