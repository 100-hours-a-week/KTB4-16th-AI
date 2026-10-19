"""자물쇠가 저장될 때 그 한 건의 조건 값을 바로 반영한다 (기능4 작업이 시작될 때)."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.components.record_facts_sync import SELECT_ONE, sync_one
from app.workers import embedding_worker
from app.workers import record_facts_worker as worker


def _row(record_id=1029):
    return SimpleNamespace(
        record_id=record_id,
        user_id=7,
        created_at=datetime(2026, 10, 20, 9, 30),
        weather_condition="CLEAR",
        temperature=18.5,
        mood_score=10,
        comment="산책",
        legal_dong_name="성수동1가",
        legal_dong_code=1120011500,
        latitude=37.544,
        longitude=127.037,
        track_title="소나기",
        artist_name="리센느",
    )


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class FakeMysql:
    def __init__(self, row):
        self.row = row
        self.calls: list[tuple[str, dict]] = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params):
        self.calls.append((str(statement), dict(params)))
        return _Result(self.row)


class FakePg:
    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeRepository:
    saved: list = []

    def __init__(self, _session):
        pass

    async def upsert_many(self, facts):
        FakeRepository.saved.extend(facts)
        return len(facts)


# ---- 한 건 읽기 --------------------------------------------------------------


async def test_one_record_is_read_by_its_id_and_saved():
    FakeRepository.saved = []
    mysql = FakeMysql(_row(1029))

    saved = await sync_one(
        mysql, FakePg(), record_id=1029, store_comment=False, repository=FakeRepository
    )

    assert saved is True
    ((sql, params),) = mysql.calls
    assert "r.record_id = :record_id" in sql and params == {"record_id": 1029}
    (fact,) = FakeRepository.saved
    assert (fact.record_id, fact.user_id, fact.track_title) == (1029, 7, "소나기")
    assert fact.comment_text is None  # 코멘트 원문은 기본 미저장


async def test_a_deleted_or_missing_record_is_not_saved():
    FakeRepository.saved = []

    saved = await sync_one(
        FakeMysql(None), FakePg(), record_id=1029, store_comment=False, repository=FakeRepository
    )

    assert saved is False and FakeRepository.saved == []  # 삭제 뒤에 다시 살아나지 않는다


def test_the_one_record_query_skips_deleted_records_and_reads_no_card_information():
    sql = str(SELECT_ONE)

    assert "r.deleted_at IS NULL" in sql and "r.record_id = :record_id" in sql
    assert "album_image_url" not in sql and "external_url" not in sql


# ---- 워커의 안전한 호출 -------------------------------------------------------


def _settings(monkeypatch, **values):
    base = {
        "record_facts_sync_enabled": True,
        "backend_mysql_url": "mysql+aiomysql://x",
        "record_facts_store_comment": False,
    }
    base.update(values)
    monkeypatch.setattr(worker, "get_settings", lambda: SimpleNamespace(**base))


@pytest.fixture
def calls(monkeypatch):
    log: list[int] = []

    async def fake_sync_one(mysql, pg, *, record_id, store_comment):
        log.append(record_id)
        return True

    monkeypatch.setattr(worker, "sync_one", fake_sync_one)
    monkeypatch.setattr(worker.mysql, "get_sessionmaker", lambda: "mysql-maker")
    monkeypatch.setattr(worker.postgres, "get_sessionmaker", lambda: "pg-maker")
    return log


async def test_record_is_synced_when_enabled(monkeypatch, calls):
    _settings(monkeypatch)

    await worker.sync_record(1029)

    assert calls == [1029]


@pytest.mark.parametrize(
    "overrides", [{"record_facts_sync_enabled": False}, {"backend_mysql_url": ""}]
)
async def test_nothing_is_read_when_disabled_or_without_a_backend(monkeypatch, calls, overrides):
    _settings(monkeypatch, **overrides)

    await worker.sync_record(1029)

    assert calls == []


async def test_a_failure_never_escapes_so_the_embedding_job_is_not_failed(monkeypatch, calls):
    _settings(monkeypatch)

    async def broken(*args, **kwargs):
        raise RuntimeError("백엔드 DB 장애")

    monkeypatch.setattr(worker, "sync_one", broken)

    await worker.sync_record(1029)  # 예외가 밖으로 나오지 않는다


# ---- 기능4 작업과의 연결 -------------------------------------------------------


async def test_the_embedding_job_syncs_the_record_first_and_still_builds_the_vectors(
    monkeypatch, record_payload
):
    order: list[str] = []

    async def fake_sync(record_id):
        order.append(f"facts:{record_id}")

    class FakeService:
        def __init__(self, **kwargs):
            pass

        async def generate(self, req):
            order.append(f"vectors:{req.record_id}")

    class FakeSession:
        def __call__(self):
            return self

        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(embedding_worker.record_facts_worker, "sync_record", fake_sync)
    monkeypatch.setattr(embedding_worker, "EmbeddingService", FakeService)
    monkeypatch.setattr(embedding_worker, "get_sessionmaker", lambda: FakeSession())
    monkeypatch.setattr(
        embedding_worker,
        "get_clients",
        lambda: SimpleNamespace(clip=None, embedding=None, llm_knowledge=None),
    )
    monkeypatch.setattr(embedding_worker, "get_category_vectors", lambda: None)
    monkeypatch.setattr(embedding_worker, "get_axis_vectors", lambda: None)
    monkeypatch.setattr(embedding_worker, "get_tag_version", lambda: None)
    monkeypatch.setattr(embedding_worker, "ClipTagger", lambda *a: None)
    monkeypatch.setattr(embedding_worker, "Embedder", lambda *a: None)
    monkeypatch.setattr(embedding_worker, "MusicMoodTagger", lambda *a: None)
    monkeypatch.setattr(embedding_worker, "TrackMoodResolver", lambda **k: None)
    monkeypatch.setattr(embedding_worker, "TrackMoodRepository", lambda s: None)
    monkeypatch.setattr(embedding_worker, "EmbeddingRepository", lambda s: None)

    await embedding_worker.handle(record_payload)

    assert order == ["facts:1029", "vectors:1029"]  # 조건 값이 먼저, 벡터는 그 뒤
