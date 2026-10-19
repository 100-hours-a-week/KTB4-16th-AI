"""record_facts 저장소 — 읽기(MemoryRecordSource와 같은 세 가지)와 쓰기가 만드는 SQL을 확인한다.

로컬·CI에 pgvector DB가 없어서 실제로 실행하지는 못한다 (실제 DB 검증은 따로 했다).
대신 어떤 표를 어떤 조건으로 읽고 쓰는지 생성되는 SQL을 확인해서 사용자 조건을 빼먹는 실수를 막는다.
"""

from datetime import datetime
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from app.db.models import RecordFact
from app.db.repositories.memory_record_repository import TrackName
from app.db.repositories.record_fact_repository import RecordFactData, RecordFactRepository


class CapturingSession:
    """실행된 문장을 기록하고 미리 정한 결과를 돌려준다."""

    def __init__(self, rows=(), scalars=()):
        self.rows = list(rows)
        self.scalar_rows = list(scalars)
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.rows, self.scalar_rows)

    async def commit(self):
        self.commits += 1


class _Result(list):
    def __init__(self, rows, scalar_rows):
        super().__init__(rows)
        self._scalars = scalar_rows

    def scalars(self):
        return iter(self._scalars)


def _compile(statement):
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


def _fact(record_id=1, **overrides):
    values = {
        "record_id": record_id,
        "user_id": 7,
        "recorded_at": datetime(2026, 10, 15, 21, 30),
        "weather_condition": "CLOUDY",
        "temperature": 14.5,
        "mood_score": -10,
        "dong_name": "성수동1가",
        "dong_code": "1120011500",
        "latitude": 37.544,
        "longitude": 127.037,
        "comment_text": None,
        "track_title": "소나기",
        "artist_name": "RESCENE, 리센느",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---- 읽기 -------------------------------------------------------------------


async def test_existing_ids_query_is_scoped_to_user():
    session = CapturingSession([SimpleNamespace(record_id=1)])

    ids = await RecordFactRepository(session).existing_ids(user_id=7, record_ids=[1, 2])

    sql, params = _compile(session.statements[0])
    assert "record_facts.user_id =" in sql  # 다른 사용자의 자물쇠는 읽히지 않는다
    assert "record_facts.record_id IN" in sql
    assert 7 in params.values()
    assert ids == {1}


async def test_no_ids_means_no_query():
    session = CapturingSession()

    assert await RecordFactRepository(session).existing_ids(user_id=7, record_ids=[]) == set()
    assert session.statements == []


async def test_condition_rows_are_this_users_newest_first_and_limited():
    session = CapturingSession(scalars=[_fact(2), _fact(1)])

    rows = await RecordFactRepository(session).list_condition_rows(user_id=7, limit=300)

    sql, params = _compile(session.statements[0])
    assert "record_facts.user_id =" in sql
    assert "ORDER BY record_facts.recorded_at DESC, record_facts.record_id DESC" in sql
    assert "LIMIT" in sql and 300 in params.values()
    assert [r.record_id for r in rows] == [2, 1]


async def test_condition_rows_have_the_same_shape_the_backend_query_gave():
    session = CapturingSession(scalars=[_fact(1, comment_text="광교산 단풍")])

    (row,) = await RecordFactRepository(session).list_condition_rows(user_id=7, limit=10)

    # 날짜·시각은 그대로(시간대 없음), 이름 매핑은 백엔드 조회(ConditionRow)와 같다
    assert row.created_at == datetime(2026, 10, 15, 21, 30) and row.created_at.tzinfo is None
    assert (row.weather_condition, row.temperature, row.mood_score) == ("CLOUDY", 14.5, -10)
    assert (row.dong_name, row.dong_code) == ("성수동1가", "1120011500")
    assert (row.latitude, row.longitude) == (37.544, 127.037)
    assert row.comment == "광교산 단풍"
    assert (row.track_title, row.artist_name) == ("소나기", "RESCENE, 리센느")


async def test_condition_rows_keep_missing_values_as_none():
    empty = _fact(
        3,
        weather_condition=None,
        temperature=None,
        mood_score=None,
        dong_name=None,
        dong_code=None,
        latitude=None,
        longitude=None,
        comment_text=None,
        track_title=None,
        artist_name=None,
    )
    session = CapturingSession(scalars=[empty])

    (row,) = await RecordFactRepository(session).list_condition_rows(user_id=7, limit=10)

    assert row.weather_condition is None and row.temperature is None and row.mood_score is None
    assert row.dong_name is None and row.latitude is None and row.comment is None
    assert row.track_title is None and row.artist_name is None


async def test_track_names_are_distinct_scoped_to_user_and_skip_rows_without_a_track():
    session = CapturingSession([SimpleNamespace(track_title="소나기", artist_name="RESCENE")])

    names = await RecordFactRepository(session).list_track_names(user_id=7, limit=500)

    sql, params = _compile(session.statements[0])
    assert "SELECT DISTINCT record_facts.track_title, record_facts.artist_name" in sql
    assert "record_facts.user_id =" in sql
    assert "track_title IS NOT NULL" in sql and "artist_name IS NOT NULL" in sql
    assert 7 in params.values() and 500 in params.values()
    assert names == [TrackName("소나기", "RESCENE")]


# ---- 쓰기 -------------------------------------------------------------------


async def test_upsert_overwrites_the_same_record_and_commits():
    session = CapturingSession()
    facts = [
        RecordFactData(1, 7, datetime(2026, 10, 15, 21, 30), weather_condition="CLOUDY"),
        RecordFactData(2, 7, datetime(2026, 10, 16, 8, 5), track_title="밤편지"),
    ]

    saved = await RecordFactRepository(session).upsert_many(facts)

    sql, _ = _compile(session.statements[0])
    assert "INSERT INTO record_facts" in sql
    assert "ON CONFLICT (record_id) DO UPDATE" in sql  # 같은 자물쇠는 1행만 (멱등)
    for column in ("recorded_at", "weather_condition", "track_title", "artist_name"):
        assert f"{column} = excluded.{column}" in sql
    assert "synced_at = now()" in sql
    assert "record_id = excluded" not in sql  # 기본키는 바꾸지 않는다
    assert saved == 2 and session.commits == 1


async def test_upsert_nothing_does_nothing():
    session = CapturingSession()

    assert await RecordFactRepository(session).upsert_many([]) == 0
    assert session.statements == [] and session.commits == 0


async def test_delete_many_removes_only_the_given_records():
    session = CapturingSession()

    await RecordFactRepository(session).delete_many([1, 2])
    await RecordFactRepository(session).delete_many([])  # 빈 목록은 아무것도 안 한다

    assert len(session.statements) == 1
    sql, _ = _compile(session.statements[0])
    assert "DELETE FROM record_facts WHERE record_facts.record_id IN" in sql
    assert session.commits == 1


def test_model_columns_match_the_migration():
    cols = RecordFact.__table__.c

    assert cols.recorded_at.type.timezone is False  # 한국시간 값을 그대로 둔다
    assert cols.comment_text.nullable is True  # 코멘트 원문은 기본으로 비운다
    assert {c.name for c in cols} == {
        "record_id",
        "user_id",
        "recorded_at",
        "weather_condition",
        "temperature",
        "mood_score",
        "dong_name",
        "dong_code",
        "latitude",
        "longitude",
        "comment_text",
        "track_title",
        "artist_name",
        "synced_at",
    }
