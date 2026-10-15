"""기능6 매일 배치 저장소 — 백엔드 MySQL(저장한 곡)을 읽고 AI Postgres에 그룹·질문을 쓴다."""

from datetime import date, timedelta

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.music_genre_assigner import Assignment, SavedTrack
from app.db.models import TrackMood, UserMusicGenre

PAST_QUESTION_DAYS = 60


class BalanceDailyRepository:
    def __init__(self, mysql_session: AsyncSession | None, postgres_session: AsyncSession):
        self._mysql = mysql_session
        self._pg = postgres_session

    async def saved_tracks(self) -> list[SavedTrack]:
        """삭제 안 된 모든 자물쇠의 (사용자, 곡, 저장 시각). 백엔드 DB가 없으면 빈 목록."""
        if self._mysql is None:
            return []
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.user_id, m.external_track_id, r.created_at
                FROM records r
                JOIN music_tracks m ON r.music_track_id = m.music_track_id
                WHERE r.deleted_at IS NULL
                """
            )
        )
        return [SavedTrack(row.user_id, row.external_track_id, row.created_at) for row in rows]

    async def genre_of(self, track_ids: set[str]) -> dict[str, str]:
        if not track_ids:
            return {}
        rows = await self._pg.execute(
            select(TrackMood.external_track_id, TrackMood.genre).where(
                TrackMood.external_track_id.in_(track_ids)
            )
        )
        return {r.external_track_id: r.genre for r in rows}

    async def save_assignments(self, assignments: dict[int, Assignment]) -> None:
        if not assignments:
            return
        stmt = insert(UserMusicGenre).values(
            [
                {"user_id": uid, "music_genre": a.music_genre, "source": a.source}
                for uid, a in assignments.items()
            ]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "music_genre": stmt.excluded.music_genre,
                "source": stmt.excluded.source,
                "updated_at": text("now()"),
            },
        )
        await self._pg.execute(stmt)
        await self._pg.commit()

    async def filled_groups(self, question_date: date) -> set[str]:
        rows = await self._pg.execute(
            text(
                "SELECT music_genre FROM vote_questions "
                "WHERE question_date = :d AND status = 'approved'"
            ),
            {"d": question_date},
        )
        return {r.music_genre for r in rows}

    async def recent_questions(self, today: date) -> set[str]:
        """최근에 낸 질문 — 같은 질문이 반복되지 않게 고를 때 뺀다."""
        rows = await self._pg.execute(
            text("SELECT question FROM vote_questions WHERE question_date >= :since"),
            {"since": today - timedelta(days=PAST_QUESTION_DAYS)},
        )
        return {r.question for r in rows}

    async def add_question(
        self,
        *,
        question_date: date,
        music_genre: str,
        question: str,
        option_a: str,
        option_b: str,
        model_version: str,
    ) -> bool:
        """승인 질문 1개 저장. 이미 승인 질문이 있는 날짜·그룹이면 False."""
        row = await self._pg.execute(
            text(
                """
                INSERT INTO vote_questions
                    (question_date, music_genre, question, option_a, option_b, model_version)
                VALUES (:d, :g, :q, :a, :b, :v)
                ON CONFLICT (question_date, music_genre) WHERE status = 'approved' DO NOTHING
                RETURNING id
                """
            ),
            {
                "d": question_date,
                "g": music_genre,
                "q": question,
                "a": option_a,
                "b": option_b,
                "v": model_version,
            },
        )
        inserted = row.first() is not None
        await self._pg.commit()
        return inserted
