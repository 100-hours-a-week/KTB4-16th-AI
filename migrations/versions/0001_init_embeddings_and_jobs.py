"""record_embeddings, ai_jobs 생성

벡터 차원은 실행 시점의 설정값(CLIP_EMBEDDING_DIM, TEXT_EMBEDDING_DIM)을 따른다.
운영 적용 후 차원을 바꾸려면 이 파일을 고치지 말고 새 마이그레이션 + 전체 재임베딩으로 처리한다.

Revision ID: 0001
Revises:
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from app.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    settings = get_settings()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "record_embeddings",
        sa.Column("record_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("user_id", sa.BigInteger, nullable=False),
        sa.Column("image_embedding", Vector(settings.clip_embedding_dim), nullable=False),
        sa.Column("music_embedding", Vector(settings.text_embedding_dim), nullable=False),
        sa.Column("comment_embedding", Vector(settings.text_embedding_dim), nullable=True),
        sa.Column("music_mood_text", sa.Text, nullable=False),
        sa.Column("model_versions", JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_record_embeddings_user_id", "record_embeddings", ["user_id"])
    for column in ("image_embedding", "music_embedding", "comment_embedding"):
        op.execute(
            f"CREATE INDEX ix_record_embeddings_{column}_hnsw "
            f"ON record_embeddings USING hnsw ({column} vector_cosine_ops)"
        )

    op.create_table(
        "ai_jobs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("job_type", sa.String(32), nullable=False),
        sa.Column("dedupe_key", sa.String(128), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer, server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer, nullable=False),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column(
            "run_after", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("job_type", "dedupe_key", name="uq_ai_jobs_type_key"),
    )
    op.create_index("ix_ai_jobs_status_run_after", "ai_jobs", ["status", "run_after"])


def downgrade() -> None:
    op.drop_table("ai_jobs")
    op.drop_table("record_embeddings")
