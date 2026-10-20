"""가수 별칭 만들기 작업 — 변환하지 않은 가수만 고르고, 실패는 다음 주기로 미룬다."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from app import worker_main
from app.components.track_lexicon import compact
from app.workers import artist_alias_worker as w


class FakeRepo:
    def __init__(self, names, known=()):
        self.names = list(names)
        self.known = set(known)
        self.saved: dict[str, tuple[str, tuple[str, ...], str]] = {}
        self.versions_asked: list[str] = []

    async def source_artist_names(self, limit):
        return self.names

    async def known_keys(self, version):
        self.versions_asked.append(version)
        return set(self.known)

    async def save(self, *, name_key, artist_name, aliases, version):
        self.saved[name_key] = (artist_name, tuple(aliases), version)


class FakeTagger:
    version = "stub|v1"

    def __init__(self, table=None, fail=(), delay=0.0):
        self.table = table or {}
        self.fail = set(fail)
        self.delay = delay
        self.calls: list[str] = []
        self.running = 0
        self.peak = 0

    async def aliases(self, name):
        self.calls.append(name)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if name in self.fail:
                raise RuntimeError("LLM 오류")
            return self.table.get(name, ())
        finally:
            self.running -= 1


async def _refresh(repo, tagger, **overrides):
    options = {"concurrency": 5, "max_per_run": 100, **overrides}
    return await w.refresh(repo, tagger, **options)


async def test_new_artists_are_converted_and_saved_under_their_compact_name():
    repo = FakeRepo(["BIGBANG", "아이유"])
    tagger = FakeTagger({"BIGBANG": ("빅뱅",), "아이유": ("IU",)})

    result = await _refresh(repo, tagger)

    assert repo.saved == {
        "bigbang": ("BIGBANG", ("빅뱅",), "stub|v1"),
        "아이유": ("아이유", ("IU",), "stub|v1"),
    }
    assert (result.converted, result.failed, result.remaining) == (2, 0, 0)


async def test_a_column_with_several_artists_is_split_and_spelling_variants_are_asked_once():
    repo = FakeRepo(["로꼬, 펀치", "BIGBANG", "Big Bang", "big-bang"])
    tagger = FakeTagger()

    await _refresh(repo, tagger)

    assert sorted(tagger.calls) == ["BIGBANG", "로꼬", "펀치"]
    assert set(repo.saved) == {"bigbang", "로꼬", "펀치"}


async def test_artists_already_converted_with_the_current_version_are_not_asked_again():
    repo = FakeRepo(["BIGBANG", "DAY6"], known={"bigbang"})
    tagger = FakeTagger()

    await _refresh(repo, tagger)

    assert tagger.calls == ["DAY6"]
    assert repo.versions_asked == ["stub|v1"]  # 현재 LLM·프롬프트 버전으로 비교한다


async def test_an_unknown_artist_is_saved_as_empty_so_it_is_not_asked_every_hour():
    repo = FakeRepo(["SunLake"])

    result = await _refresh(repo, FakeTagger())

    assert repo.saved["sunlake"][1] == () and result.converted == 1


async def test_a_failed_call_is_not_saved_and_does_not_stop_the_others():
    repo = FakeRepo(["A-Team", "BIGBANG", "DAY6"])
    tagger = FakeTagger({"BIGBANG": ("빅뱅",), "DAY6": ("데이식스",)}, fail={"A-Team"})

    result = await _refresh(repo, tagger)

    assert set(repo.saved) == {"bigbang", "day6"}  # 실패한 가수는 다음 주기에 다시 시도
    assert (result.converted, result.failed) == (2, 1)


async def test_names_without_letters_are_skipped():
    repo = FakeRepo(["-", "...", "", "10CM"])
    tagger = FakeTagger()

    await _refresh(repo, tagger)

    assert tagger.calls == ["10CM"]


async def test_one_run_converts_at_most_max_per_run_and_reports_the_rest():
    repo = FakeRepo([f"Artist {i}" for i in range(7)])
    tagger = FakeTagger()

    result = await _refresh(repo, tagger, max_per_run=5)

    assert len(repo.saved) == 5 and result.remaining == 2


async def test_nothing_to_do_makes_no_llm_call():
    repo = FakeRepo(["BIGBANG"], known={"bigbang"})
    tagger = FakeTagger()

    result = await _refresh(repo, tagger)

    assert tagger.calls == [] and repo.saved == {}
    assert (result.converted, result.failed, result.remaining) == (0, 0, 0)


async def test_calls_run_in_parallel_but_not_beyond_the_concurrency_limit():
    repo = FakeRepo([f"Artist {i}" for i in range(12)])
    tagger = FakeTagger(delay=0.01)

    await _refresh(repo, tagger, concurrency=3)

    assert 1 < tagger.peak <= 3 and len(repo.saved) == 12


async def test_results_are_saved_in_chunks_so_a_stop_midway_loses_little():
    repo = FakeRepo([f"Artist {i}" for i in range(5)])
    tagger = FakeTagger()

    await w.refresh(repo, tagger, concurrency=5, max_per_run=100, chunk_size=2)

    assert len(repo.saved) == 5 and compact("Artist 0") in repo.saved


def _settings(**values):
    base = {
        "artist_alias_enabled": True,
        "artist_alias_interval_minutes": 60,
        "ai_client_mode": "real",
    }
    return SimpleNamespace(**{**base, **values})


async def test_handle_does_nothing_when_disabled_or_in_fake_mode(monkeypatch):
    def forbidden():
        raise AssertionError("LLM 클라이언트를 만들면 안 된다")

    monkeypatch.setattr(w, "get_clients", forbidden)
    for values in ({"artist_alias_enabled": False}, {"ai_client_mode": "fake"}):
        monkeypatch.setattr(w, "get_settings", lambda values=values: _settings(**values))

        await w.handle({})


def test_one_job_per_interval_slot(monkeypatch):
    monkeypatch.setattr(w, "get_settings", lambda: _settings())

    same_hour = (
        w.slot_key(datetime(2026, 10, 20, 3, 7, tzinfo=UTC)),
        w.slot_key(datetime(2026, 10, 20, 3, 59, tzinfo=UTC)),
    )
    assert same_hour[0] == same_hour[1]
    assert w.slot_key(datetime(2026, 10, 20, 4, 0, tzinfo=UTC)) != same_hour[0]
    fields = w.job_fields(datetime(2026, 10, 20, 3, 7, tzinfo=UTC))
    assert fields["job_type"] == "artist_alias" and fields["max_attempts"] == 1


def test_the_handler_is_registered_with_the_worker():
    assert worker_main.HANDLERS[w.JOB_TYPE] is w.handle


async def test_scheduling_is_skipped_when_disabled_and_done_once_per_slot_when_enabled(monkeypatch):
    called = []

    async def fake_schedule(now):
        called.append(now)

    monkeypatch.setattr(w, "schedule", fake_schedule)
    monkeypatch.setattr(worker_main, "_alias_slots", set())
    now = datetime(2026, 10, 20, 5, 0, tzinfo=UTC)

    monkeypatch.setattr(worker_main, "get_settings", lambda: _settings(artist_alias_enabled=False))
    await worker_main.schedule_artist_alias(now)
    assert called == []

    monkeypatch.setattr(w, "get_settings", lambda: _settings())
    monkeypatch.setattr(worker_main, "get_settings", lambda: _settings())
    await worker_main.schedule_artist_alias(now)
    await worker_main.schedule_artist_alias(now)  # 같은 칸에서는 한 번만
    assert called == [now]


def test_defaults_are_on_hourly_and_capped(monkeypatch):
    from app.config import Settings

    for name in (
        "ARTIST_ALIAS_ENABLED",
        "ARTIST_ALIAS_INTERVAL_MINUTES",
        "ARTIST_ALIAS_CONCURRENCY",
        "ARTIST_ALIAS_MAX_PER_RUN",
    ):
        monkeypatch.delenv(name, raising=False)

    s = Settings(_env_file=None)

    assert s.artist_alias_enabled is True and s.artist_alias_interval_minutes == 60
    assert s.artist_alias_concurrency == 5 and s.artist_alias_max_per_run == 100
