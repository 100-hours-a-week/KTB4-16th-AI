"""텍스트 임베딩을 OpenAI(1536차원)에서 KURE-v1(1024차원)로 바꾼다

벡터 칸의 차원이 달라지므로 기존 텍스트 벡터는 옮길 수 없다. 그래서 칸을 1024차원으로 바꾸면서
기존 값을 비운다(NULL). 비워진 값은 `python -m app.tools.reembed_text`로 다시 채운다.
- record_embeddings.music_embedding / comment_embedding: 곡 무드 글·코멘트의 벡터
- track_moods.mood_embedding: 곡 무드 글의 벡터 (무드 글 자체와 장르·무드 태그는 그대로 둔다)
사진 벡터(image_embedding, CLIP)는 건드리지 않는다.

재임베딩이 끝나기 전에는 비어 있는 행이 생기므로 NOT NULL을 푼다. 값을 비우기 **전에** 풀어야 한다
(행이 있는 DB에서 NOT NULL 칸을 NULL로 바꾸면 위반이라 순서가 바뀌면 실패한다).
비어 있는 행은 기억 검색·기능1 검색에서 빠지고(코사인 거리가 NULL), track_moods는 캐시로 쓰지 않고
새로 만든다.

HNSW 인덱스는 칸을 바꾸기 전에 지우고 바꾼 뒤 다시 만든다 (record_embeddings 두 칸).

되돌리기(downgrade)도 값을 비운다. 1536차원으로 돌아가도 KURE 벡터를 옮길 수 없고, OpenAI로
다시 임베딩해야 한다. NOT NULL은 복원하지 않는다(값이 빈 행이 있을 수 있어서).

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-16
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

KURE_DIM = 1024
OPENAI_DIM = 1536
INDEXED_COLUMNS = ("music_embedding", "comment_embedding")


def _retype(dim: int) -> None:
    for column in INDEXED_COLUMNS:
        op.execute(f"DROP INDEX IF EXISTS ix_record_embeddings_{column}_hnsw")
    # NOT NULL을 먼저 푼다. 칸 종류를 바꾸면서 값을 NULL로 비우는데(USING NULL), 그 뒤에 풀면
    # 행이 하나라도 있는 DB에서는 NOT NULL 위반으로 실패한다 (2026-10-17 개발 서버에서 겪음).
    # 행이 없는 DB에서는 순서와 상관없이 통과해서 로컬 시험에서는 드러나지 않았다.
    for table, column in (
        ("record_embeddings", "music_embedding"),
        ("track_moods", "mood_embedding"),
    ):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL")
    for table, column in (
        ("record_embeddings", "music_embedding"),
        ("record_embeddings", "comment_embedding"),
        ("track_moods", "mood_embedding"),
    ):
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE vector({dim}) USING NULL::vector({dim})"
        )
    for column in INDEXED_COLUMNS:
        op.execute(
            f"CREATE INDEX ix_record_embeddings_{column}_hnsw "
            f"ON record_embeddings USING hnsw ({column} vector_cosine_ops)"
        )


def upgrade() -> None:
    _retype(KURE_DIM)


def downgrade() -> None:
    _retype(OPENAI_DIM)
