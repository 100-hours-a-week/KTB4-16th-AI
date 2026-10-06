"""track_lookups 추가 — LLM 추천곡의 Spotify 확인 결과 캐시

Spotify 하루 요청 한도 초과로 추천이 전부 멈춘 뒤(2026-10-02), 한 번 확인한 곡은
다시 조회하지 않도록 결과를 저장한다. 유효기간은 코드(track_resolver)에서 관리한다.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "track_lookups",
        sa.Column("lookup_key", sa.String(300), primary_key=True),
        sa.Column("llm_artist", sa.String(200), nullable=False),
        sa.Column("llm_title", sa.String(300), nullable=False),
        sa.Column("genre", sa.String(20), nullable=False, server_default=""),
        sa.Column("moods", sa.ARRAY(sa.String(20)), nullable=False, server_default="{}"),
        sa.Column("external_track_id", sa.String(64), nullable=True),
        sa.Column("title", sa.String(300), nullable=True),
        sa.Column("artist_name", sa.String(300), nullable=True),
        sa.Column("spotify_uri", sa.String(100), nullable=True),
        sa.Column("album_image_url", sa.Text, nullable=True),
        sa.Column("external_url", sa.Text, nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_track_lookups_external_track_id", "track_lookups", ["external_track_id"])


def downgrade() -> None:
    op.drop_index("ix_track_lookups_external_track_id", table_name="track_lookups")
    op.drop_table("track_lookups")
