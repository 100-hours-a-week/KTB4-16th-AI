"""monthly_reports.photo_scenes를 JSONB로 변경

백엔드 콜백 DTO가 photoScenes를 [{"tag","count","ratio"}, ...] 구조로
요구해서, 태그 이름만 담던 문자열 배열로는 부족하다. 아직 실제로 쌓인
운영 데이터가 없어(0004가 막 배포된 시점) 값 보존 없이 컬럼을 새로 만든다.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("monthly_reports", "photo_scenes")
    op.add_column(
        "monthly_reports", sa.Column("photo_scenes", JSONB, server_default="[]", nullable=False)
    )


def downgrade() -> None:
    op.drop_column("monthly_reports", "photo_scenes")
    op.add_column(
        "monthly_reports",
        sa.Column("photo_scenes", sa.ARRAY(sa.String(20)), server_default="{}", nullable=False),
    )
