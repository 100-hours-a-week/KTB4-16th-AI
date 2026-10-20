import io
import re

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db.models import RecordEmbedding

# alembic.ini에 path_separator가 없다는 alembic 안내 — 이 시험과 무관하다
pytestmark = pytest.mark.filterwarnings("ignore:No path_separator found")


def _script() -> ScriptDirectory:
    return ScriptDirectory.from_config(Config("alembic.ini"))


def test_migrations_have_a_single_head():
    """브랜치 두 개가 같은 번호로 마이그레이션을 추가하면 head가 둘이 되어 서버가 못 뜬다."""
    assert len(_script().get_heads()) == 1


def test_photo_tags_migration_follows_the_balance_daily_migration():
    """0008은 기능6(오늘의 질문 배치)이 먼저 썼다. 번호가 겹쳐서 이쪽을 0009로 올렸다."""
    revision = _script().get_revision("0009")

    assert revision.down_revision == "0008"


def test_text_embedding_migration_follows_the_photo_tags_migration():
    revision = _script().get_revision("0010")

    assert revision.down_revision == "0009"


def test_text_vector_columns_match_the_kure_dimension_and_can_be_empty():
    """0010이 칸을 1024차원으로 바꾸고 NOT NULL을 푼다 — 모델 정의가 DB와 같아야 한다."""
    from app.db.models import TrackMood

    for column in (
        RecordEmbedding.__table__.c.music_embedding,
        RecordEmbedding.__table__.c.comment_embedding,
        TrackMood.__table__.c.mood_embedding,
    ):
        assert column.type.dim == 1024
        assert column.nullable is True
    assert RecordEmbedding.__table__.c.image_embedding.type.dim == 768  # 사진(CLIP)은 그대로


def test_photo_tags_is_stored_as_sql_null_when_empty():
    """JSONB는 기본으로 None을 JSON null로 저장한다 — NULL(분류 안 함)과 구분하려면 켜야 한다."""
    assert RecordEmbedding.__table__.c.photo_tags.type.none_as_null is True
    assert RecordEmbedding.__table__.c.photo_tags.nullable is True


def _offline_sql(revisions: str) -> list[str]:
    """DB 없이 마이그레이션이 실행할 SQL을 순서대로 뽑는다 (alembic의 --sql 모드와 같다)."""
    buffer = io.StringIO()
    # 설정 파일(alembic.ini)을 주지 않는다: 주면 migrations/env.py가 로그 설정을 다시 읽어서
    # 이후 시험의 로그 확인(caplog)이 빈 값이 된다.
    config = Config(output_buffer=buffer)
    config.set_main_option("script_location", "migrations")
    start, end = revisions.split(":")
    if start < end:
        command.upgrade(config, revisions, sql=True)
    else:
        command.downgrade(config, revisions, sql=True)
    statements = [s.strip() for s in buffer.getvalue().split(";")]
    return [re.sub(r"\s+", " ", s) for s in statements if s]


@pytest.mark.parametrize("revisions", ["0009:0010", "0010:0009"])
def test_text_vector_columns_drop_not_null_before_their_type_changes(revisions):
    """NOT NULL 칸의 값을 비우는(USING NULL) 변경은, 그 전에 NOT NULL을 풀지 않으면 행이 있는 DB에서
    실패한다.

    개발 서버에서 이 순서가 거꾸로라 마이그레이션이 실패했다 (2026-10-17). 행이 없는 로컬 DB에서는
    통과해서 드러나지 않았으므로, 실행될 SQL의 순서를 직접 확인한다.
    """
    sql = _offline_sql(revisions)

    for table, column in (
        ("record_embeddings", "music_embedding"),
        ("track_moods", "mood_embedding"),
    ):
        alter = f"ALTER TABLE {table} ALTER COLUMN {column}"
        drop = next(i for i, s in enumerate(sql) if f"{alter} DROP NOT NULL" in s)
        retype = next(i for i, s in enumerate(sql) if f"{alter} TYPE vector" in s)

        assert drop < retype, f"{table}.{column}: NOT NULL을 풀기 전에 칸 종류를 바꾸면 안 된다"


def test_text_vector_indexes_are_rebuilt_after_the_type_changes():
    """칸 종류를 바꾸기 전에 HNSW 인덱스를 지우고, 바꾼 뒤에 다시 만든다."""
    sql = _offline_sql("0009:0010")

    retypes = [i for i, s in enumerate(sql) if "ALTER COLUMN" in s and "TYPE vector" in s]
    drops = [i for i, s in enumerate(sql) if s.startswith("DROP INDEX")]
    creates = [i for i, s in enumerate(sql) if s.startswith("CREATE INDEX")]

    assert drops and all(i < min(retypes) for i in drops)
    assert creates and all(i > max(retypes) for i in creates)


def test_record_facts_migration_follows_the_text_embedding_migration_and_only_adds_a_table():
    revision = _script().get_revision("0011")

    assert revision.down_revision == "0010"
    sql = _offline_sql("0010:0011")
    assert any("CREATE TABLE record_facts" in s for s in sql)
    assert any("CREATE INDEX ix_record_facts_user_recorded" in s for s in sql)
    # 기존 표는 건드리지 않는다 (값은 채우기 도구가 넣는다)
    assert not any("ALTER TABLE" in s for s in sql)
    down = _offline_sql("0011:0010")
    assert any("DROP TABLE record_facts" in s for s in down)


def test_sync_state_migration_follows_the_record_facts_migration():
    revision = _script().get_revision("0012")

    assert revision.down_revision == "0011"
