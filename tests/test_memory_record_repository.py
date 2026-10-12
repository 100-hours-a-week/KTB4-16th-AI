from datetime import datetime
from decimal import Decimal
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


def _condition_row(record_id=1, **overrides):
    values = {
        "record_id": record_id,
        "created_at": datetime(2026, 10, 3, 18, 30),
        "weather_condition": "CLEAR",
        "temperature": Decimal("18.5"),
        "mood_score": 10,
        "legal_dong_name": "성수동1가",
        "legal_dong_code": "1120011500",
        "latitude": Decimal("37.544000"),
        "longitude": Decimal("127.037000"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def test_condition_rows_query_is_scoped_to_user_and_limited():
    session = FakeMysqlSession([_condition_row(1)])

    await MemoryRecordRepository(session).list_condition_rows(user_id=5, limit=300)

    sql, params = session.executed[0]
    assert "r.user_id = :user_id" in sql  # 다른 사용자의 자물쇠는 읽히지 않는다
    assert "r.deleted_at IS NULL" in sql  # 삭제된 자물쇠는 뺀다
    assert "ORDER BY r.created_at DESC" in sql and "LIMIT :limit" in sql  # 최근순, 개수 제한
    assert params == {"user_id": 5, "limit": 300}


async def test_condition_rows_convert_decimals_and_codes():
    session = FakeMysqlSession([_condition_row(7, legal_dong_code=1120011500)])

    (row,) = await MemoryRecordRepository(session).list_condition_rows(user_id=5, limit=10)

    assert (row.record_id, row.weather_condition, row.mood_score) == (7, "CLEAR", 10)
    assert row.temperature == 18.5 and isinstance(row.temperature, float)
    assert (row.latitude, row.longitude) == (37.544, 127.037)
    assert row.dong_name == "성수동1가"
    assert row.dong_code == "1120011500"  # 숫자로 와도 문자열로


async def test_condition_rows_keep_missing_values_as_none():
    empty = _condition_row(
        2,
        weather_condition=None,
        temperature=None,
        mood_score=None,
        legal_dong_name=None,
        legal_dong_code=None,
        latitude=None,
        longitude=None,
    )
    session = FakeMysqlSession([empty])

    (row,) = await MemoryRecordRepository(session).list_condition_rows(user_id=5, limit=10)

    assert row.weather_condition is None and row.temperature is None and row.mood_score is None
    assert row.dong_name is None and row.dong_code is None
    assert row.latitude is None and row.longitude is None


async def test_get_many_does_not_read_the_unverified_place_columns():
    """위경도·법정동코드 컬럼 이름이 틀려도 조건이 없는 질문의 검색이 깨지지 않게 분리해 둔다."""
    session = FakeMysqlSession([_row(1)])

    await MemoryRecordRepository(session).get_many(user_id=5, record_ids=[1])

    sql, _ = session.executed[0]
    assert "latitude" not in sql and "longitude" not in sql and "legal_dong_code" not in sql
