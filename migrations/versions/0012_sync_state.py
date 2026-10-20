"""기능4 자물쇠 동기화 위치 — 백엔드 DB에서 어디까지 읽었는지 기억한다

백엔드가 자물쇠 저장 때 AI를 호출하는 대신, AI 워커가 5분마다 백엔드 DB에서 바뀐
자물쇠를 읽어 온다(pull). 마지막으로 읽은 (updated_at, record_id)를 여기 저장한다.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sync_state",
        sa.Column("name", sa.String(50), primary_key=True),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_record_id", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_table("sync_state")
