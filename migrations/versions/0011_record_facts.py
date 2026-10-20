"""record_facts 표 추가 — 기억 검색이 읽는 자물쇠 조건 값의 사본

기억 검색은 지금 날짜·날씨·기온·기분·장소·곡명·가수 같은 값을 질문마다 백엔드 MySQL에서 읽는다.
이 값을 AI PostgreSQL에 두고 거기서 읽을 수 있게 표를 만든다 (자물쇠 1건 = 1행).
- recorded_at: 백엔드 records.created_at 그대로. 한국시간 값이라 시간대 없는 칸에 둔다
  (시간대·요일·날짜 조건이 이 값에서 시·요일·날짜를 꺼내므로 바꾸지 않는다)
- comment_text: 사전에 없는 장소를 코멘트 글자로 찾을 때만 쓴다. 기본은 비워 둔다 (개인정보)
- track_title·artist_name: 곡명·가수 검색용. 백엔드 music_tracks의 값
행은 채우기 도구(app/tools/backfill_record_facts.py)가 넣는다. 이 마이그레이션은 표만 만든다.
읽는 곳은 설정(MEMORY_RECORD_SOURCE)으로 바꾸고, 기본은 지금 방식(mysql)이다.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "record_facts",
        sa.Column("record_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("user_id", sa.BigInteger, nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("weather_condition", sa.String(30), nullable=True),
        sa.Column("temperature", sa.Float(53), nullable=True),
        sa.Column("mood_score", sa.SmallInteger, nullable=True),
        sa.Column("dong_name", sa.String(100), nullable=True),
        sa.Column("dong_code", sa.String(10), nullable=True),
        sa.Column("latitude", sa.Float(53), nullable=True),
        sa.Column("longitude", sa.Float(53), nullable=True),
        sa.Column("comment_text", sa.Text, nullable=True),
        sa.Column("track_title", sa.String(300), nullable=True),
        sa.Column("artist_name", sa.String(300), nullable=True),
        sa.Column(
            "synced_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_record_facts_user_recorded", "record_facts", ["user_id", "recorded_at"])


def downgrade() -> None:
    op.drop_index("ix_record_facts_user_recorded", table_name="record_facts")
    op.drop_table("record_facts")
