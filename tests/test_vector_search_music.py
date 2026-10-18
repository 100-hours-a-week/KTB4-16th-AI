"""곡 분위기 벡터 검색이 만드는 SQL 확인.

로컬·CI에 pgvector DB가 없어서 실제로 실행하지는 못한다. 대신 어떤 컬럼을 어떤 연산으로 비교하고
어떤 사용자로 거르는지, 생성되는 SQL을 확인한다 (컬럼을 잘못 쓰거나 사용자 조건을 빼먹는 실수 방지).
"""

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


async def test_music_search_compares_music_embedding_for_this_user_only():
    session = CapturingSession()

    await VectorSearch(session).similar_by_music(user_id=7, query_vector=[0.1, 0.2, 0.3], limit=20)

    sql, params = _compile(session.statements[0])
    assert "record_embeddings.music_embedding <=>" in sql  # 코사인 거리, 곡 분위기 벡터
    assert "record_embeddings.user_id =" in sql  # 다른 사용자의 기록은 후보에도 안 든다
    assert "ORDER BY" in sql and "LIMIT" in sql
    assert 7 in params.values() and 20 in params.values()


async def test_music_search_skips_records_whose_vector_is_not_re_embedded_yet():
    """모델 교체(0010) 뒤 재임베딩 전의 행은 벡터가 비어 있다 — 순위에 끼지 않게 뺀다."""
    session = CapturingSession()

    await VectorSearch(session).similar_by_music(user_id=7, query_vector=[0.1], limit=5)

    sql, _ = _compile(session.statements[0])
    assert "record_embeddings.music_embedding IS NOT NULL" in sql


async def test_music_search_converts_distance_to_similarity():
    from types import SimpleNamespace

    rows = [SimpleNamespace(record_id=3, external_track_id="abc", distance=0.25)]

    hits = await VectorSearch(CapturingSession(rows)).similar_by_music(
        user_id=7, query_vector=[0.1], limit=5
    )

    assert [(h.record_id, h.external_track_id, h.similarity) for h in hits] == [(3, "abc", 0.75)]


async def test_comment_search_still_skips_records_without_comment_vector():
    """기능1이 쓰는 기존 코멘트 검색은 그대로다 — 코멘트가 없는 기록은 비교 대상이 아니다."""
    session = CapturingSession()

    await VectorSearch(session).similar_by_comment(user_id=7, query_vector=[0.1], limit=5)

    sql, _ = _compile(session.statements[0])
    assert "record_embeddings.comment_embedding <=>" in sql
    assert "record_embeddings.comment_embedding IS NOT NULL" in sql
