"""track_moods에 곡 제목·가수 칸 추가

지금까지는 Spotify 곡 ID로만 곡을 식별했다. 제목·가수를 같이 저장하면 추천할 때 Spotify를
다시 부르지 않고도 곡 풀(가수·제목으로 식별)과 같은 곡인지 비교할 수 있다.
- NULL: 이 칸이 생기기 전에 저장된 곡. 같은 곡이 다시 저장될 때(기능4) 채워진다.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("track_moods", sa.Column("title", sa.String(300), nullable=True))
    op.add_column("track_moods", sa.Column("artist_name", sa.String(300), nullable=True))


def downgrade() -> None:
    op.drop_column("track_moods", "artist_name")
    op.drop_column("track_moods", "title")
