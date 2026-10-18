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


def _row(record_id=1):
    return SimpleNamespace(record_id=record_id)


async def test_ids_query_is_scoped_to_user_and_skips_deleted_records():
    session = FakeMysqlSession([_row(1), _row(2)])

    await MemoryRecordRepository(session).existing_ids(user_id=5, record_ids=[1, 2])

    sql, params = session.executed[0]
    assert "r.user_id = :user_id" in sql  # 다른 사용자의 자물쇠는 읽히지 않는다
    assert "r.deleted_at IS NULL" in sql  # 삭제된 자물쇠는 뺀다
    assert params == {"user_id": 5, "record_ids": [1, 2]}


async def test_existing_ids_returns_only_the_ids_the_database_found():
    # DB가 1번만 돌려주면(2번은 삭제됐거나 다른 사용자 것) 1번만 남는다
    session = FakeMysqlSession([_row(1)])

    ids = await MemoryRecordRepository(session).existing_ids(user_id=5, record_ids=[1, 2])

    assert ids == {1}


async def test_ids_query_reads_no_card_information():
    """응답은 ID만 준다 — 곡·장소 테이블이나 카드 컬럼을 읽지 않는다 (명세 5장)."""
    session = FakeMysqlSession([_row(1)])

    await MemoryRecordRepository(session).existing_ids(user_id=5, record_ids=[1])

    sql, _ = session.executed[0]
    for card_part in ("music_tracks", "places", "title", "artist_name", "album_image_url"):
        assert card_part not in sql
    assert "SELECT r.record_id" in sql


async def test_a_record_without_a_place_or_music_row_is_still_found():
    """예전에는 곡·장소와 JOIN해서 그 정보가 없는 자물쇠가 빠졌다. ID만 확인하니 빠지지 않는다."""
    session = FakeMysqlSession([_row(9)])

    ids = await MemoryRecordRepository(session).existing_ids(user_id=5, record_ids=[9])

    assert ids == {9}
    assert "JOIN" not in session.executed[0][0]


async def test_no_ids_means_no_query():
    session = FakeMysqlSession([_row(1)])

    assert await MemoryRecordRepository(session).existing_ids(user_id=5, record_ids=[]) == set()
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
        "comment": "광교산 단풍",
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
    assert "r.comment" in sql  # 글자 검색에 쓰는 코멘트
    assert "music_tracks" not in sql  # 곡 제목·가수 이름은 글자 검색 대상이 아니다


async def test_condition_rows_convert_decimals_and_codes():
    session = FakeMysqlSession([_condition_row(7, legal_dong_code=1120011500)])

    (row,) = await MemoryRecordRepository(session).list_condition_rows(user_id=5, limit=10)

    assert (row.record_id, row.weather_condition, row.mood_score) == (7, "CLEAR", 10)
    assert row.temperature == 18.5 and isinstance(row.temperature, float)
    assert (row.latitude, row.longitude) == (37.544, 127.037)
    assert row.dong_name == "성수동1가"
    assert row.dong_code == "1120011500"  # 숫자로 와도 문자열로
    assert row.comment == "광교산 단풍"


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
        comment=None,
    )
    session = FakeMysqlSession([empty])

    (row,) = await MemoryRecordRepository(session).list_condition_rows(user_id=5, limit=10)

    assert row.weather_condition is None and row.temperature is None and row.mood_score is None
    assert row.dong_name is None and row.dong_code is None
    assert row.latitude is None and row.longitude is None
    assert row.comment is None  # 코멘트를 안 쓴 자물쇠
