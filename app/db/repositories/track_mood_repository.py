"""track_moods 접근 — 곡 단위 무드 묘사 캐시.

이 테이블은 사용자가 자물쇠에 저장한 곡마다 쌓이므로, 시간이 지나면 "실제로 사람들이
고른 곡 + 장르·무드 태그" 목록이 된다. 기능1·3 추천이 곡 풀(TJ 차트 기반 고정 목록)과
같은 후보 풀로 함께 쓴다 (search_by_tags).
"""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TrackMood


@dataclass(frozen=True)
class CachedMood:
    mood_text: str
    mood_embedding: list[float]
    genre: str
    moods: tuple[str, ...]
    title: str | None = None  # 이 칸이 생기기 전에 저장된 곡은 None
    artist_name: str | None = None


@dataclass(frozen=True)
class SavedSong:
    """사용자들이 저장한 곡 — 추천 후보로 쓸 때 필요한 값만."""

    external_track_id: str
    genre: str
    moods: tuple[str, ...]
    title: str | None = None
    artist_name: str | None = None
    # 분위기 문장의 벡터 — 상황 문장과 가까운 순으로 줄 세울 때 쓴다 (재임베딩 전엔 None)
    embedding: tuple[float, ...] | None = None


class SavedSongSearch(Protocol):
    async def search_by_tags(
        self, genres: list[str], moods: list[str], limit: int
    ) -> list[SavedSong]: ...


class TrackMoodStore(Protocol):
    async def get(self, external_track_id: str, mood_version: str) -> CachedMood | None: ...

    async def save(self, external_track_id: str, mood_version: str, mood: CachedMood) -> None: ...


class TrackMoodRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, external_track_id: str, mood_version: str) -> CachedMood | None:
        row = (
            await self._session.execute(
                select(
                    TrackMood.mood_text,
                    TrackMood.mood_embedding,
                    TrackMood.genre,
                    TrackMood.moods,
                    TrackMood.title,
                    TrackMood.artist_name,
                ).where(
                    TrackMood.external_track_id == external_track_id,
                    TrackMood.mood_version == mood_version,
                    # 모델 교체(0010) 뒤 재임베딩 전의 행(NULL)은 캐시로 쓰지 않는다
                    TrackMood.mood_embedding.is_not(None),
                )
            )
        ).first()
        if row is None:
            return None
        return CachedMood(
            mood_text=row.mood_text,
            mood_embedding=list(row.mood_embedding),
            genre=row.genre,
            moods=tuple(row.moods),
            title=row.title,
            artist_name=row.artist_name,
        )

    async def save(self, external_track_id: str, mood_version: str, mood: CachedMood) -> None:
        stmt = insert(TrackMood).values(
            external_track_id=external_track_id,
            mood_text=mood.mood_text,
            mood_embedding=mood.mood_embedding,
            genre=mood.genre,
            moods=list(mood.moods),
            mood_version=mood_version,
            title=mood.title,
            artist_name=mood.artist_name,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[TrackMood.external_track_id],
            set_={
                "mood_text": stmt.excluded.mood_text,
                "mood_embedding": stmt.excluded.mood_embedding,
                "genre": stmt.excluded.genre,
                "moods": stmt.excluded.moods,
                "mood_version": stmt.excluded.mood_version,
                "title": stmt.excluded.title,
                "artist_name": stmt.excluded.artist_name,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def search_by_tags(
        self, genres: list[str], moods: list[str], limit: int
    ) -> list[SavedSong]:
        """장르가 같거나 무드가 하나라도 겹치는 저장된 곡. 최근에 쓰인 것부터.

        점수·선별은 호출하는 쪽(saved_song_pool)이 곡 풀과 같은 기준으로 한다 —
        여기서는 후보만 넉넉히 넘긴다. 옛 어휘(13장르 이전)로 붙은 태그는 애초에
        지금 장르·무드 목록과 안 맞아서 자연히 걸러진다.
        """
        if not genres and not moods:
            return []
        conditions = []
        if genres:
            conditions.append(TrackMood.genre.in_(genres))
        if moods:
            # moods 칸은 일반 ARRAY로 선언돼 있어 .overlap()이 없다 — 겹침 연산자를 직접 쓴다
            conditions.append(TrackMood.moods.bool_op("&&")(cast(moods, PG_ARRAY(String(20)))))
        rows = await self._session.execute(
            select(
                TrackMood.external_track_id,
                TrackMood.genre,
                TrackMood.moods,
                TrackMood.title,
                TrackMood.artist_name,
                TrackMood.mood_embedding,
            )
            .where(or_(*conditions))
            .order_by(TrackMood.updated_at.desc())
            .limit(limit)
        )
        return [
            SavedSong(
                external_track_id=row.external_track_id,
                genre=row.genre,
                moods=tuple(row.moods),
                title=row.title,
                artist_name=row.artist_name,
                embedding=tuple(row.mood_embedding) if row.mood_embedding is not None else None,
            )
            for row in rows
        ]
