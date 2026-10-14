from datetime import datetime
from types import SimpleNamespace

from app.db.repositories.memory_record_repository import MemoryRecordRepository


class FakeMysqlSession:
    """실행된 SQL과 파라미터를 기록하고 미리 정한 행을 돌려준다."""

    def __init__(self, rows):
        self.rows = rows
        self.executed: list[tuple[str, dict]] = []

    async def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return list(self.rows)


def _row(record_id=1, **overrides):
    values = {
        "record_id": record_id,
        "created_at": datetime(2026, 8, 24, 18, 30),
        "title": "밤편지",
        "artist_name": "아이유",
        "external_track_id": "abc",
        "album_image_url": "https://img/cover.jpg",
        "external_url": "https://open.spotify.com/track/abc",
        "place_id": 10,
        "place_name": "오산동",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def test_query_is_scoped_to_user_and_skips_deleted_records():
    session = FakeMysqlSession([_row(1), _row(2)])

    await MemoryRecordRepository(session).get_many(user_id=5, record_ids=[1, 2])

    sql, params = session.executed[0]
    assert "r.user_id = :user_id" in sql  # 다른 사용자의 자물쇠는 읽히지 않는다
    assert "r.deleted_at IS NULL" in sql  # 삭제된 자물쇠는 뺀다
    assert params == {"user_id": 5, "record_ids": [1, 2]}


async def test_rows_are_mapped_by_record_id():
    session = FakeMysqlSession([_row(1), _row(2, title="Blueming", place_name=None)])

    records = await MemoryRecordRepository(session).get_many(user_id=5, record_ids=[1, 2])

    assert set(records) == {1, 2}
    assert records[2].title == "Blueming"
    assert records[2].place_name is None


async def test_missing_image_or_link_becomes_empty_string_not_none():
    session = FakeMysqlSession([_row(1, album_image_url=None, external_url=None)])

    records = await MemoryRecordRepository(session).get_many(user_id=5, record_ids=[1])

    assert records[1].album_image_url == ""
    assert records[1].external_url == ""


async def test_no_ids_means_no_query():
    session = FakeMysqlSession([_row(1)])

    assert await MemoryRecordRepository(session).get_many(user_id=5, record_ids=[]) == {}
    assert session.executed == []
