"""record_facts 동기화 작업 예약과 실행 — 이름(구간), 켜짐·꺼짐, 종류별 호출."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app import worker_main
from app.workers import record_facts_worker as worker


def _settings(monkeypatch, **values):
    base = {
        "record_facts_sync_enabled": True,
        "backend_mysql_url": "mysql+aiomysql://x",
        "record_facts_sync_interval_seconds": 300,
        "record_facts_sync_lookback_minutes": 60,
        "record_facts_full_sync_hour": 4,
        "record_facts_store_comment": False,
    }
    base.update(values)
    monkeypatch.setattr(worker, "get_settings", lambda: SimpleNamespace(**base))


def _utc(hour, minute=0, day=19):
    # 한국시간은 UTC보다 9시간 빠르다 (UTC 10:07 = 한국 19:07)
    return datetime(2026, 10, day, hour, minute, 30, tzinfo=UTC)


# ---- 작업 이름 ---------------------------------------------------------------


def test_recent_key_is_the_same_within_five_minutes_and_changes_after(monkeypatch):
    _settings(monkeypatch)

    assert worker.recent_key(_utc(10, 7)) == "recent:2026-10-19T19:05"
    assert worker.recent_key(_utc(10, 9)) == "recent:2026-10-19T19:05"
    assert worker.recent_key(_utc(10, 10)) == "recent:2026-10-19T19:10"


def test_recent_key_uses_korea_time(monkeypatch):
    _settings(monkeypatch)

    assert worker.recent_key(_utc(15, 0)) == "recent:2026-10-20T00:00"  # 한국은 다음 날 0시


def test_full_key_exists_only_after_the_configured_korea_hour(monkeypatch):
    _settings(monkeypatch, record_facts_full_sync_hour=4)

    assert worker.full_key(_utc(18, 59)) is None  # 한국 03:59
    assert worker.full_key(_utc(19, 0)) == "full:2026-10-20"  # 한국 04:00
    assert worker.full_key(_utc(23, 0)) == "full:2026-10-20"  # 한국 08:00, 같은 날은 같은 이름


# ---- 예약 --------------------------------------------------------------------


class FakeJobs:
    enqueued: list[dict] = []
    known: set[tuple[str, str]] = set()

    def __init__(self, _session):
        pass

    async def enqueue_if_absent(self, *, job_type, dedupe_key, payload, max_attempts):
        FakeJobs.enqueued.append(
            {"job_type": job_type, "key": dedupe_key, "payload": payload, "max": max_attempts}
        )
        first = (job_type, dedupe_key) not in FakeJobs.known
        FakeJobs.known.add((job_type, dedupe_key))
        return first


class _Maker:
    def __call__(self):
        return self

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def jobs(monkeypatch):
    FakeJobs.enqueued = []
    FakeJobs.known = set()
    monkeypatch.setattr(worker, "JobRepository", FakeJobs)
    monkeypatch.setattr(worker.postgres, "get_sessionmaker", lambda: _Maker())
    return FakeJobs


async def test_the_current_five_minute_window_is_queued_once(monkeypatch, jobs):
    _settings(monkeypatch)
    last: dict[str, str] = {}

    await worker.enqueue_due(_utc(10, 7), last)
    await worker.enqueue_due(_utc(10, 8), last)  # 같은 구간 — 다시 넣지 않는다
    await worker.enqueue_due(_utc(10, 11), last)  # 다음 구간

    # 한국시간 19시는 전체 맞추기 시각(4시) 뒤라서 full 작업도 하루 한 번 같이 들어간다
    recent = [e for e in jobs.enqueued if e["payload"]["mode"] == "recent"]
    assert [e["key"] for e in recent] == ["recent:2026-10-19T19:05", "recent:2026-10-19T19:10"]
    assert recent[0]["payload"] == {"mode": "recent"}
    assert recent[0]["job_type"] == "record_facts_sync"


async def test_the_daily_full_sync_is_queued_once_a_day_after_the_hour(monkeypatch, jobs):
    _settings(monkeypatch)
    last: dict[str, str] = {}

    await worker.enqueue_due(_utc(19, 1), last)  # 한국 04:01
    await worker.enqueue_due(_utc(19, 6), last)  # 같은 날, 다음 구간

    fulls = [e for e in jobs.enqueued if e["payload"]["mode"] == "full"]
    assert [e["key"] for e in fulls] == ["full:2026-10-20"]


async def test_two_servers_queue_the_same_window_only_once(monkeypatch, jobs):
    _settings(monkeypatch)

    first, second = {}, {}  # 서버 두 대의 메모리
    await worker.enqueue_due(_utc(10, 7), first)
    await worker.enqueue_due(_utc(10, 7), second)

    # 서버마다 큐에 넣으려 하지만 큐의 유니크 키가 두 번째를 받지 않는다 (가짜 큐도 같은 규칙)
    recent = [e for e in jobs.enqueued if e["payload"]["mode"] == "recent"]
    assert [e["key"] for e in recent] == ["recent:2026-10-19T19:05"] * 2
    assert len({(e["job_type"], e["key"]) for e in jobs.enqueued}) == 2  # recent 하나, full 하나


@pytest.mark.parametrize(
    "overrides",
    [
        {"record_facts_sync_enabled": False},  # 꺼짐 (기본)
        {"backend_mysql_url": ""},  # 백엔드 DB 주소 없음
    ],
)
async def test_nothing_is_queued_when_disabled_or_without_a_backend(monkeypatch, jobs, overrides):
    _settings(monkeypatch, **overrides)

    await worker.enqueue_due(_utc(10, 7), {})

    assert jobs.enqueued == []


# ---- 실행 --------------------------------------------------------------------


class Called:
    calls: list[tuple[str, dict]] = []


async def _fake_sync_recent(mysql, pg, **kwargs):
    Called.calls.append(("recent", kwargs))
    return SimpleNamespace(upserted=2, deleted=1)


async def _fake_reconcile(mysql, pg, **kwargs):
    Called.calls.append(("full", kwargs))
    return SimpleNamespace(upserted=5, deleted=0)


class FakeCleanupJobs:
    cleaned: list[tuple[str, int]] = []

    def __init__(self, _session):
        pass

    async def delete_finished(self, *, job_type, older_than_hours):
        FakeCleanupJobs.cleaned.append((job_type, older_than_hours))
        return 0


@pytest.fixture
def runs(monkeypatch):
    Called.calls = []
    FakeCleanupJobs.cleaned = []
    monkeypatch.setattr(worker, "sync_recent", _fake_sync_recent)
    monkeypatch.setattr(worker, "reconcile", _fake_reconcile)
    monkeypatch.setattr(worker, "JobRepository", FakeCleanupJobs)
    monkeypatch.setattr(worker.mysql, "get_sessionmaker", lambda: "mysql-maker")
    monkeypatch.setattr(worker.postgres, "get_sessionmaker", lambda: _Maker())
    return Called.calls


async def test_a_recent_job_reads_back_the_lookback_window_and_cleans_old_jobs(monkeypatch, runs):
    _settings(monkeypatch, record_facts_sync_lookback_minutes=60)

    await worker.handle({"mode": "recent"})

    ((mode, kwargs),) = runs
    assert mode == "recent"
    age = worker.now_kst() - kwargs["since"]
    assert 59 * 60 <= age.total_seconds() <= 61 * 60  # 60분 전부터 읽는다
    assert kwargs["store_comment"] is False
    assert FakeCleanupJobs.cleaned == [("record_facts_sync", 48)]  # 끝난 작업은 이틀 뒤 정리


async def test_a_full_job_reconciles_everything(monkeypatch, runs):
    _settings(monkeypatch, record_facts_store_comment=True)

    await worker.handle({"mode": "full"})

    ((mode, kwargs),) = runs
    assert mode == "full" and kwargs["store_comment"] is True


async def test_a_job_that_runs_while_disabled_does_nothing(monkeypatch, runs):
    _settings(monkeypatch, record_facts_sync_enabled=False)

    await worker.handle({"mode": "recent"})

    assert runs == [] and FakeCleanupJobs.cleaned == []


# ---- worker_main 연결 --------------------------------------------------------


def test_the_worker_knows_how_to_run_record_facts_sync_jobs():
    assert worker_main.HANDLERS["record_facts_sync"] is worker.handle


async def test_a_scheduling_failure_does_not_stop_the_worker_loop(monkeypatch):
    async def broken(now, last):
        raise RuntimeError("DB 장애")

    monkeypatch.setattr(worker_main.record_facts_worker, "enqueue_due", broken)

    await worker_main.schedule_record_facts_sync(_utc(10, 7))  # 예외가 밖으로 나오지 않는다
