"""장르를 온보딩과 같은 13개로 — 목록에 없는 예전 장르 태그는 비운다

곡 장르 어휘를 멜론·지니·벅스 공통 13개로 바꾸면서(app/components/music_tags.py),
예전 17개 중 13개에 없는 이름(시티팝, 로파이, 칠, 어쿠스틱, 클래식, 트랩)과
이름이 바뀐 것(힙합, 알앤비, 락, 인디, 팝, 일렉트로닉, 포크)으로 저장된 태그를 비운다.
비운 곡은 다음에 무드를 다시 만들 때 새 목록으로 채워진다.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GENRES = (
    "발라드", "댄스", "랩/힙합", "R&B/Soul", "인디음악", "록/메탈", "포크/블루스",
    "트로트", "POP", "일렉트로니카", "OST", "재즈", "J-POP",
)  # fmt: skip


def upgrade() -> None:
    allowed = ", ".join(f"'{g}'" for g in GENRES)
    for table in ("track_moods", "track_lookups"):
        op.execute(f"UPDATE {table} SET genre = '' WHERE genre <> '' AND genre NOT IN ({allowed})")


def downgrade() -> None:
    # 지운 태그는 되살릴 수 없다 (다음 무드 생성 때 새로 채워짐)
    pass
