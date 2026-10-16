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
