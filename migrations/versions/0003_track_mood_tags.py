"""track_moods에 genre·moods 컬럼 추가

기능3 재랭킹을 임베딩 유사도에서 장르·무드 태그 겹침으로 바꾸면서 필요해졌다.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "track_moods", sa.Column("genre", sa.String(20), nullable=False, server_default="")
    )
    op.add_column(
        "track_moods",
        sa.Column("moods", sa.ARRAY(sa.String(20)), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("track_moods", "moods")
    op.drop_column("track_moods", "genre")
