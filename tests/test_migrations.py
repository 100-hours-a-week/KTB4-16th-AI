import pytest
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


def test_photo_tags_migration_follows_the_previous_head():
    revision = _script().get_revision("0008")

    assert revision.down_revision == "0007"


def test_photo_tags_is_stored_as_sql_null_when_empty():
    """JSONB는 기본으로 None을 JSON null로 저장한다 — NULL(분류 안 함)과 구분하려면 켜야 한다."""
    assert RecordEmbedding.__table__.c.photo_tags.type.none_as_null is True
    assert RecordEmbedding.__table__.c.photo_tags.nullable is True
