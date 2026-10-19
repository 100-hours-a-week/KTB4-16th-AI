"""조건 검색이 쓰는 두 조회의 SQL 확인 — 곡 무드 태그 읽기, 지정한 기록들의 유사도.

로컬·CI에 pgvector DB가 없어서 실제로 실행하지는 못한다. 대신 어떤 테이블을 어떤 조건으로
읽는지 생성되는 SQL을 확인한다 (사용자 조건을 빼먹거나 컬럼을 잘못 쓰는 실수 방지).
"""

from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from app.components.vector_search import VectorSearch


class CapturingSession:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.rows


def _compile(statement):
    compiled = statement.compile(dialect=postgresql.dialect())
    return str(compiled), compiled.params


async def test_moods_are_read_through_the_track_id_for_this_user_only():
    session = CapturingSession()

    await VectorSearch(session).moods_by_record(user_id=7)

    sql, params = _compile(session.statements[0])
    assert (
        "JOIN track_moods ON record_embeddings.external_track_id = track_moods.external_track_id"
        in sql
    )
    assert "record_embeddings.user_id =" in sql  # 다른 사용자의 기록은 읽지 않는다
    assert 7 in params.values()


async def test_moods_map_record_to_tags_and_missing_tags_become_empty():
    rows = [
        SimpleNamespace(record_id=1, moods=["잔잔한", "차분한"]),
        SimpleNamespace(record_id=2, moods=[]),
        SimpleNamespace(record_id=3, moods=None),
    ]

    result = await VectorSearch(CapturingSession(rows)).moods_by_record(user_id=7)

    assert result == {1: ("잔잔한", "차분한"), 2: (), 3: ()}


async def test_genres_are_read_through_the_track_id_for_this_user_only():
    session = CapturingSession()

    await VectorSearch(session).genres_by_record(user_id=7)

    sql, params = _compile(session.statements[0])
    assert "track_moods.genre" in sql
    assert (
        "JOIN track_moods ON record_embeddings.external_track_id = track_moods.external_track_id"
        in sql
    )
    assert "record_embeddings.user_id =" in sql  # 다른 사용자의 기록은 읽지 않는다
    assert 7 in params.values()


async def test_genres_map_record_to_tag_and_skip_records_without_a_genre():
    rows = [
        SimpleNamespace(record_id=1, genre="발라드"),
        SimpleNamespace(record_id=2, genre=""),
        SimpleNamespace(record_id=3, genre=None),
    ]

    result = await VectorSearch(CapturingSession(rows)).genres_by_record(user_id=7)

    assert result == {1: "발라드"}


async def test_similarities_for_reads_only_the_given_records_of_this_user():
    session = CapturingSession()

    await VectorSearch(session).similarities_for(
        user_id=7, record_ids=[3, 4, 5], query_vector=[0.1, 0.2]
    )

    comment_sql, _ = _compile(session.statements[0])
    music_sql, params = _compile(session.statements[1])
    assert "record_embeddings.comment_embedding <=>" in comment_sql
    assert "record_embeddings.comment_embedding IS NOT NULL" in comment_sql
    assert "record_embeddings.music_embedding <=>" in music_sql
    for sql in (comment_sql, music_sql):
        assert "record_embeddings.user_id =" in sql
        assert "record_embeddings.record_id IN" in sql
        assert "LIMIT" not in sql  # 지정한 기록은 전부 필요하다
    assert 7 in params.values()


async def test_similarities_for_converts_distance_and_returns_comment_then_music():
    rows = [SimpleNamespace(record_id=3, external_track_id="abc", distance=0.25)]

    comment, music = await VectorSearch(CapturingSession(rows)).similarities_for(
        user_id=7, record_ids=[3], query_vector=[0.1]
    )

    assert [(h.record_id, h.similarity) for h in comment] == [(3, 0.75)]
    assert [(h.record_id, h.similarity) for h in music] == [(3, 0.75)]


async def test_similarities_for_no_records_runs_no_query():
    session = CapturingSession()

    result = await VectorSearch(session).similarities_for(
        user_id=7, record_ids=[], query_vector=[0.1]
    )

    assert result == ([], [])
    assert session.statements == []
