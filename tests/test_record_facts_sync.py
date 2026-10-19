"""record_facts 동기화 — 최근 변경 반영, 전체 맞추기(사라진 행 정리), 사용하는 SQL."""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from app.components.record_facts_sync import (
    SELECT_ACTIVE,
    SELECT_DELETED_SINCE,
    SELECT_RECENT,
    SELECT_STILL_ACTIVE,
    now_kst,
    reconcile,
    sync_recent,
)


def _row(record_id, **overrides):
    values = {
        "record_id": record_id,
        "user_id": 7,
        "created_at": datetime(2026, 10, 19, 21, 30),
        "weather_condition": "CLOUDY",
        "temperature": Decimal("14.5"),
        "mood_score": -10,
        "comment": "광교산 단풍",
        "legal_dong_name": "성수동1가",
        "legal_dong_code": 1120011500,
        "latitude": Decimal("37.544"),
        "longitude": Decimal("127.037"),
        "track_title": "소나기",
        "artist_name": "리센느",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Rows(list):
    def all(self):
        return list(self)


class FakeMysql:
    """백엔드 MySQL 대역. 문장에 따라 미리 정한 행을 돌려주고 받은 파라미터를 기록한다."""

    def __init__(self, *, active=(), recent=(), deleted=(), alive=None):
        self.active = list(active)
        self.recent = list(recent)
        self.deleted = list(deleted)
        self.alive = alive  # None이면 active의 ID
        self.calls: list[tuple[str, dict]] = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params):
        sql = str(statement)
        self.calls.append((sql, dict(params)))
        if "IN :ids" in sql or "IN (" in sql:
            alive = self.alive if self.alive is not None else {r.record_id for r in self.active}
            return _Rows(SimpleNamespace(record_id=i) for i in params["ids"] if i in alive)
        if "deleted_at IS NOT NULL" in sql:
            rows = [r for r in self.deleted if r > params["after_id"]][: params["limit"]]
            return _Rows(SimpleNamespace(record_id=r) for r in rows)
        source = self.recent if "created_at >= :since" in sql else self.active
        return _Rows([r for r in source if r.record_id > params["after_id"]][: params["limit"]])


class FakePg:
    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeRepository:
    """record_facts 저장소 대역. 저장된 행과 삭제 호출을 기록한다."""

    rows: dict[int, object] = {}
    deleted: list[list[int]] = []

    def __init__(self, _session):
        pass

    @classmethod
    def reset(cls, existing_ids=()):
        cls.rows = {i: object() for i in existing_ids}
        cls.deleted = []

    async def upsert_many(self, facts):
        for f in facts:
            FakeRepository.rows[f.record_id] = f
        return len(facts)

    async def delete_many(self, record_ids):
        FakeRepository.deleted.append(list(record_ids))
        for i in record_ids:
            FakeRepository.rows.pop(i, None)

    async def list_record_ids(self, *, after_id, limit):
        return [i for i in sorted(FakeRepository.rows) if i > after_id][:limit]


SINCE = datetime(2026, 10, 19, 20, 30)


# ---- 최근 변경 ---------------------------------------------------------------


async def test_recent_sync_saves_recent_records_and_removes_recently_deleted_ones():
    FakeRepository.reset(existing_ids=[5, 6])
    mysql = FakeMysql(recent=[_row(10), _row(11)], deleted=[5])

    counts = await sync_recent(
        mysql, FakePg(), since=SINCE, batch_size=10, store_comment=False, repository=FakeRepository
    )

    assert (counts.upserted, counts.deleted) == (2, 1)
    assert sorted(FakeRepository.rows) == [6, 10, 11]  # 5번은 지워지고 6번은 그대로
    assert FakeRepository.deleted == [[5]]


async def test_recent_sync_asks_only_for_records_since_the_given_time():
    FakeRepository.reset()
    mysql = FakeMysql(recent=[_row(1)])

    await sync_recent(
        mysql, FakePg(), since=SINCE, batch_size=10, store_comment=False, repository=FakeRepository
    )

    assert all(params["since"] == SINCE for _, params in mysql.calls)
    assert any("r.created_at >= :since" in sql for sql, _ in mysql.calls)
    assert any(
        "r.deleted_at IS NOT NULL AND r.deleted_at >= :since" in sql for sql, _ in mysql.calls
    )


async def test_recent_sync_reads_in_batches_and_saves_everything():
    FakeRepository.reset()
    mysql = FakeMysql(recent=[_row(i) for i in range(1, 6)])

    counts = await sync_recent(
        mysql, FakePg(), since=SINCE, batch_size=2, store_comment=False, repository=FakeRepository
    )

    assert counts.upserted == 5 and sorted(FakeRepository.rows) == [1, 2, 3, 4, 5]
    cursors = [p["after_id"] for sql, p in mysql.calls if "created_at >= :since" in sql]
    assert cursors == [0, 2, 4, 5]


async def test_syncing_the_same_records_twice_changes_nothing():
    FakeRepository.reset()
    mysql = FakeMysql(recent=[_row(1), _row(2)])

    for _ in range(2):
        await sync_recent(
            mysql,
            FakePg(),
            since=SINCE,
            batch_size=10,
            store_comment=False,
            repository=FakeRepository,
        )

    assert sorted(FakeRepository.rows) == [1, 2]  # 겹쳐 읽어도 한 행씩만


async def test_recent_sync_keeps_the_comment_out_unless_asked():
    FakeRepository.reset()

    await sync_recent(
        FakeMysql(recent=[_row(1)]),
        FakePg(),
        since=SINCE,
        batch_size=10,
        store_comment=False,
        repository=FakeRepository,
    )
    assert FakeRepository.rows[1].comment_text is None

    await sync_recent(
        FakeMysql(recent=[_row(1)]),
        FakePg(),
        since=SINCE,
        batch_size=10,
        store_comment=True,
        repository=FakeRepository,
    )
    assert FakeRepository.rows[1].comment_text == "광교산 단풍"


# ---- 전체 맞추기 -------------------------------------------------------------


async def test_reconcile_saves_every_active_record_and_removes_rows_that_disappeared():
    FakeRepository.reset(existing_ids=[1, 2, 3, 99])  # 99번은 백엔드에 없다 (완전 삭제)
    mysql = FakeMysql(active=[_row(1), _row(2), _row(3, track_title="새 곡")])

    counts = await reconcile(
        mysql, FakePg(), batch_size=2, store_comment=False, repository=FakeRepository
    )

    assert counts.upserted == 3 and counts.deleted == 1
    assert sorted(FakeRepository.rows) == [1, 2, 3]
    assert FakeRepository.rows[3].track_title == "새 곡"  # 수정된 값도 반영된다


async def test_reconcile_removes_rows_of_records_deleted_in_the_backend():
    FakeRepository.reset(existing_ids=[1, 2])
    # 2번은 백엔드에 행은 있지만 삭제 표시가 있다 (아직 살아 있는 ID 목록에 없다)
    mysql = FakeMysql(active=[_row(1)], alive={1})

    counts = await reconcile(
        mysql, FakePg(), batch_size=10, store_comment=False, repository=FakeRepository
    )

    assert counts.deleted == 1 and sorted(FakeRepository.rows) == [1]


async def test_reconcile_on_an_empty_backend_clears_the_table():
    FakeRepository.reset(existing_ids=[1, 2, 3])

    counts = await reconcile(
        FakeMysql(active=[]), FakePg(), batch_size=2, store_comment=False, repository=FakeRepository
    )

    assert counts.upserted == 0 and counts.deleted == 3 and FakeRepository.rows == {}


# ---- SQL ---------------------------------------------------------------------


def test_the_queries_only_read_active_records_and_no_card_information():
    for statement in (SELECT_ACTIVE, SELECT_RECENT):
        sql = str(statement)
        assert "r.deleted_at IS NULL" in sql  # 삭제된 자물쇠는 옮기지 않는다
        assert "ORDER BY r.record_id" in sql and "LIMIT :limit" in sql
        assert "LEFT JOIN places" in sql and "LEFT JOIN music_tracks" in sql
        assert "album_image_url" not in sql and "external_url" not in sql
    assert "r.created_at >= :since" in str(SELECT_RECENT)
    assert "deleted_at IS NOT NULL" in str(SELECT_DELETED_SINCE)
    assert "deleted_at IS NULL" in str(SELECT_STILL_ACTIVE)


def test_now_in_korea_time_has_no_timezone_like_the_backend_columns():
    value = now_kst()

    assert value.tzinfo is None
    assert abs((value - datetime.now()).total_seconds()) < 9 * 3600 + 60  # 한국시간 값이다
