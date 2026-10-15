"""record_embeddings에 사진 태그(photo_tags) 칸 추가

자물쇠를 저장할 때 사진을 분류한 결과(장소 + 다축 태그)를 같이 저장한다.
예: {"장소": "카페", "사진 종류": "진짜 사진", "대상": "여러 사람", "분위기": "밤의 따뜻한 조명"}
- NULL: 아직 분류하지 않았거나(이 칸이 생기기 전에 저장된 행) 분류에 실패한 행
- {}: 분류했지만 자신 있게 붙일 태그가 없는 행
어떤 태그 목록으로 분류했는지는 model_versions의 tagVersion에 남긴다.
기존 행은 NULL로 두고 값을 채우지 않는다(저장된 이미지 벡터로 나중에 다시 계산할 수 있다).

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("record_embeddings", sa.Column("photo_tags", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("record_embeddings", "photo_tags")
