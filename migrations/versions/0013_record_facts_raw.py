"""record_facts에 자물쇠 원문 전체(raw_record) 칸 추가

record_facts는 기억 검색이 쓰는 값만 칸으로 나눠 담는다. 백엔드 records의 나머지 칸까지 모두
원문 그대로 남기려고 JSON 한 칸에 통째로 담는다 (칸 목록이 바뀌어도 이 칸은 그대로 받는다).
- NULL: 이 칸이 생기기 전에 저장된 행 (다음 동기화가 채운다)

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("record_facts", sa.Column("raw_record", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("record_facts", "raw_record")
