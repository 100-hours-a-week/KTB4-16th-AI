"""track_moods 추가, record_embeddings.external_track_id 추가

곡 단위 무드 묘사 캐시. 같은 곡(스포티파이 곡 ID)은 LLM을 한 번만 부른다.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from app.config import get_settings

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    settings = get_settings()
    op.create_table(
        "track_moods",
        sa.Column("external_track_id", sa.String(64), primary_key=True),
        sa.Column("mood_text", sa.Text, nullable=False),
        sa.Column("mood_embedding", Vector(settings.text_embedding_dim), nullable=False),
        sa.Column("mood_version", sa.String(200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.add_column("record_embeddings", sa.Column("external_track_id", sa.String(64), nullable=True))
    op.create_index(
        "ix_record_embeddings_external_track_id", "record_embeddings", ["external_track_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_record_embeddings_external_track_id", "record_embeddings")
    op.drop_column("record_embeddings", "external_track_id")
    op.drop_table("track_moods")
