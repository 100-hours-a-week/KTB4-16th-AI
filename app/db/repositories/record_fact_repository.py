"""record_facts 접근 — 기억 검색이 읽는 자물쇠 조건 값 (AI PostgreSQL).

읽기: MemoryRecordSource와 같은 세 가지(existing_ids, list_condition_rows, list_track_names)라서
백엔드 MySQL 대신 이 저장소를 꽂으면 기억 검색 코드는 그대로 동작한다.
쓰기: 채우기 도구(app/tools/backfill_record_facts.py)와 이후 동기화가 쓴다. 같은 recordId가 다시
와도 1행만 유지한다 (멱등).

사용자 격리: 읽기는 모두 user_id 조건을 건다. 삭제된 자물쇠는 쓰는 쪽이 이 표에서 지운다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RecordFact
from app.db.repositories.memory_record_repository import ConditionRow, TrackName


@dataclass(frozen=True)
class RecordFactData:
    record_id: int
    user_id: int
    recorded_at: datetime  # 백엔드 records.created_at 그대로 (한국시간, 시간대 없음)
    weather_condition: str | None = None
    temperature: float | None = None
    mood_score: int | None = None
    dong_name: str | None = None
    dong_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    comment_text: str | None = None
    track_title: str | None = None
    artist_name: str | None = None


_UPDATED = (
    "user_id",
    "recorded_at",
    "weather_condition",
    "temperature",
    "mood_score",
    "dong_name",
    "dong_code",
    "latitude",
    "longitude",
    "comment_text",
    "track_title",
    "artist_name",
)


class RecordFactRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    # ---- 읽기 (MemoryRecordSource) ------------------------------------------------

    async def existing_ids(self, *, user_id: int, record_ids: list[int]) -> set[int]:
        """record_ids 중 이 사용자의 자물쇠 ID만 돌려준다 (삭제된 것은 이 표에 없다)."""
        if not record_ids:
            return set()
        rows = await self._session.execute(
            select(RecordFact.record_id).where(
                RecordFact.user_id == user_id, RecordFact.record_id.in_(record_ids)
            )
        )
        return {row.record_id for row in rows}

    async def list_condition_rows(self, *, user_id: int, limit: int) -> list[ConditionRow]:
        """이 사용자의 자물쇠를 최근순으로 limit개까지 (백엔드 MySQL에서 읽던 것과 같은 모양)."""
        rows = await self._session.execute(
            select(RecordFact)
            .where(RecordFact.user_id == user_id)
            .order_by(RecordFact.recorded_at.desc(), RecordFact.record_id.desc())
            .limit(limit)
        )
        return [_to_condition_row(fact) for fact in rows.scalars()]

    async def list_track_names(self, *, user_id: int, limit: int) -> list[TrackName]:
        """이 사용자가 저장한 곡의 제목·가수 (중복 제거). 둘 다 없는 행은 뺀다."""
        rows = await self._session.execute(
            select(RecordFact.track_title, RecordFact.artist_name)
            .where(
                RecordFact.user_id == user_id,
                or_(RecordFact.track_title.is_not(None), RecordFact.artist_name.is_not(None)),
            )
            .distinct()
            .limit(limit)
        )
        return [TrackName(title=row.track_title, artist_name=row.artist_name) for row in rows]

    # ---- 쓰기 ---------------------------------------------------------------------

    async def upsert_many(self, facts: Sequence[RecordFactData]) -> int:
        """같은 recordId가 이미 있으면 값을 덮어쓴다 (멱등). 저장한 건수를 돌려준다."""
        if not facts:
            return 0
        values = [
            {"record_id": f.record_id, **{name: getattr(f, name) for name in _UPDATED}}
            for f in facts
        ]
        stmt = insert(RecordFact).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[RecordFact.record_id],
            set_={**{name: stmt.excluded[name] for name in _UPDATED}, "synced_at": func.now()},
        )
        await self._session.execute(stmt)
        await self._session.commit()
        return len(facts)

    async def delete_many(self, record_ids: Sequence[int]) -> None:
        """없는 recordId여도 오류 없이 끝난다 (멱등)."""
        if not record_ids:
            return
        await self._session.execute(delete(RecordFact).where(RecordFact.record_id.in_(record_ids)))
        await self._session.commit()


def _to_condition_row(fact: RecordFact) -> ConditionRow:
    return ConditionRow(
        record_id=fact.record_id,
        created_at=fact.recorded_at,
        weather_condition=fact.weather_condition,
        temperature=fact.temperature,
        mood_score=fact.mood_score,
        dong_name=fact.dong_name,
        dong_code=fact.dong_code,
        latitude=fact.latitude,
        longitude=fact.longitude,
        comment=fact.comment_text,
        track_title=fact.track_title,
        artist_name=fact.artist_name,
    )
