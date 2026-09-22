"""track_moods 접근 — 곡 단위 무드 묘사 캐시."""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TrackMood


@dataclass(frozen=True)
class CachedMood:
    mood_text: str
    mood_embedding: list[float]


class TrackMoodStore(Protocol):
    async def get(self, external_track_id: str, mood_version: str) -> CachedMood | None: ...

    async def get_any(self, external_track_id: str) -> CachedMood | None: ...

    async def save(self, external_track_id: str, mood_version: str, mood: CachedMood) -> None: ...


class TrackMoodRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, external_track_id: str, mood_version: str) -> CachedMood | None:
        row = (
            await self._session.execute(
                select(TrackMood.mood_text, TrackMood.mood_embedding).where(
                    TrackMood.external_track_id == external_track_id,
                    TrackMood.mood_version == mood_version,
                )
            )
        ).first()
        if row is None:
            return None
        return CachedMood(mood_text=row.mood_text, mood_embedding=list(row.mood_embedding))

    async def get_any(self, external_track_id: str) -> CachedMood | None:
        """버전 상관없이 캐시된 무드가 있으면 반환 — 재랭킹처럼 참고용 신호로만
        쓸 때. 임베딩 모델이 바뀌지 않는 한 오래된 프롬프트로 만든 벡터도
        같은 좌표계라 재랭킹 신호로는 충분하다(기능4처럼 정본으로 저장하는
        게 아니므로 버전 불일치를 엄격히 따지지 않는다)."""
        row = (
            await self._session.execute(
                select(TrackMood.mood_text, TrackMood.mood_embedding).where(
                    TrackMood.external_track_id == external_track_id
                )
            )
        ).first()
        if row is None:
            return None
        return CachedMood(mood_text=row.mood_text, mood_embedding=list(row.mood_embedding))

    async def save(self, external_track_id: str, mood_version: str, mood: CachedMood) -> None:
        stmt = insert(TrackMood).values(
            external_track_id=external_track_id,
            mood_text=mood.mood_text,
            mood_embedding=mood.mood_embedding,
            mood_version=mood_version,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[TrackMood.external_track_id],
            set_={
                "mood_text": stmt.excluded.mood_text,
                "mood_embedding": stmt.excluded.mood_embedding,
                "mood_version": stmt.excluded.mood_version,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()
