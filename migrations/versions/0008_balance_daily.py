"""기능6 매일 배치 결과 — 사용자별 취향 그룹, 그룹별 오늘의 질문, 백엔드용 뷰

백엔드는 원본 테이블이 아니라 뷰 두 개만 읽기 전용 계정으로 조회한다. 그래서 테이블 구조를
바꿔도 뷰 모양만 지키면 백엔드 쿼리가 깨지지 않는다. 계정 생성·권한 부여는 비밀번호가
필요해 마이그레이션에 넣지 않고 클라우드 팀이 scripts/sql/be_readonly_role.sql로 실행한다.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_music_genres",
        sa.Column("user_id", sa.BigInteger, primary_key=True, autoincrement=False),
        sa.Column("music_genre", sa.String(50), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "vote_questions",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("question_date", sa.Date, nullable=False),
        sa.Column("music_genre", sa.String(50), nullable=False),
        sa.Column("question", sa.String(100), nullable=False),
        sa.Column("option_a", sa.String(50), nullable=False),
        sa.Column("option_b", sa.String(50), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="approved"),
        sa.Column("model_version", sa.String(200), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_vote_questions_date_genre", "vote_questions", ["question_date", "music_genre"]
    )
    # 날짜·그룹마다 approved 질문은 하나 — 배치가 두 번 돌거나 서버가 2대여도 중복이 안 생긴다
    op.execute(
        "CREATE UNIQUE INDEX uq_vote_questions_approved ON vote_questions "
        "(question_date, music_genre) WHERE status = 'approved'"
    )
    op.execute(
        "CREATE VIEW be_user_music_genres AS "
        "SELECT user_id, music_genre, updated_at FROM user_music_genres"
    )
    op.execute(
        "CREATE VIEW be_vote_questions AS "
        "SELECT question_date, music_genre, question, option_a, option_b "
        "FROM vote_questions WHERE status = 'approved'"
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS be_vote_questions")
    op.execute("DROP VIEW IF EXISTS be_user_music_genres")
    op.drop_table("vote_questions")
    op.drop_table("user_music_genres")
