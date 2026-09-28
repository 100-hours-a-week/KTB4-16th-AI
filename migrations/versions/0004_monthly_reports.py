"""monthly_reports 생성

기능2 RECAP 결과 저장용. 백엔드가 조회 API로 가져갈 스냅샷을 담는다.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "monthly_reports",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.BigInteger, nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("month", sa.Integer, nullable=False),
        sa.Column("ai_recap_text", sa.Text, nullable=False),
        sa.Column("photo_scenes", sa.ARRAY(sa.String(20)), server_default="{}", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("user_id", "year", "month", name="uq_monthly_reports_user_year_month"),
    )
    op.create_index("ix_monthly_reports_user_id", "monthly_reports", ["user_id"])


def downgrade() -> None:
    op.drop_table("monthly_reports")
