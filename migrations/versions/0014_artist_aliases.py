"""artist_aliases 표 추가 — 가수 이름의 다른 표기(영어↔한글)

스포티파이에는 한국 가수도 영어로("RESCENE") 저장되는 경우가 많아서, 한글로 묻는 질문이
그 가수를 못 찾는다. 워커가 record_facts의 가수 이름을 LLM으로 다른 언어 표기로 바꿔 이 표에 쌓고,
기억 검색이 질문에서 가수를 찾을 때 함께 비교한다.
- name_key: 가수 이름에서 공백·기호·대소문자를 뺀 글자 (같은 가수를 한 행으로)
- aliases: 다른 표기 목록. 빈 목록은 "LLM이 모른다고 답함"
- alias_version: 만든 LLM·프롬프트 버전. 현재 설정과 다르면 다시 만든다

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "artist_aliases",
        sa.Column("name_key", sa.String(200), primary_key=True),
        sa.Column("artist_name", sa.String(200), nullable=False),
        sa.Column("aliases", ARRAY(sa.String(100)), nullable=False, server_default="{}"),
        sa.Column("alias_version", sa.String(200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_table("artist_aliases")
