"""오늘의 질문 재료 — 최근 자물쇠에서 장르별로 많이 저장된 가수·무드·장소·날씨를 센다.

자물쇠 원본(가수·장소·날씨)은 백엔드 MySQL, 곡 장르·무드는 AI Postgres(track_moods)에
있어서 두 DB에서 따로 읽어 곡 ID로 붙인다. 사용자 정보(누가 저장했는지, 코멘트)는
LLM에 보내지 않으므로 읽지 않는다.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.components.music_tags import COMMON_GROUP
from app.db.models import TrackMood

TOP_N = 8


@dataclass(frozen=True)
class GenreStats:
    record_count: int
    top_artists: list[tuple[str, int]]
    top_moods: list[tuple[str, int]]
    top_places: list[tuple[str, int]]
    top_weathers: list[tuple[str, int]]

    @classmethod
    def empty(cls) -> "GenreStats":
        return cls(0, [], [], [], [])


class GenreStatsSource(Protocol):
    async def get(self, music_genre: str) -> GenreStats: ...


class BalanceStatsRepository:
    def __init__(self, mysql_session: AsyncSession, postgres_session: AsyncSession):
        self._mysql = mysql_session
        self._postgres = postgres_session

    async def get(self, music_genre: str, *, days: int, limit: int) -> GenreStats:
        rows = (
            await self._mysql.execute(
                text(
                    """
                    SELECT m.external_track_id, m.artist_name,
                           p.legal_dong_name AS place_name, r.weather_condition
                    FROM records r
                    JOIN music_tracks m ON r.music_track_id = m.music_track_id
                    JOIN places p ON r.place_id = p.place_id
                    WHERE r.deleted_at IS NULL
                      AND r.created_at >= NOW() - INTERVAL :days DAY
                    ORDER BY r.created_at DESC
                    LIMIT :limit
                    """
                ),
                {"days": days, "limit": limit},
            )
        ).all()
        if not rows:
            return GenreStats.empty()

        track_ids = {r.external_track_id for r in rows}
        moods = {
            t.external_track_id: t
            for t in (
                await self._postgres.execute(
                    select(TrackMood.external_track_id, TrackMood.genre, TrackMood.moods).where(
                        TrackMood.external_track_id.in_(track_ids)
                    )
                )
            ).all()
        }
        return aggregate(rows, moods, music_genre)


def aggregate(rows, moods, music_genre: str) -> GenreStats:
    """공통 그룹은 전체 자물쇠, 장르 그룹은 그 장르 곡이 담긴 자물쇠만 센다."""
    artists, mood_tags, places, weathers = Counter(), Counter(), Counter(), Counter()
    count = 0
    for r in rows:
        mood = moods.get(r.external_track_id)
        if music_genre != COMMON_GROUP and (mood is None or mood.genre != music_genre):
            continue
        count += 1
        if r.artist_name:
            artists[r.artist_name] += 1
        if mood is not None:
            mood_tags.update(mood.moods)
        if r.place_name:
            places[r.place_name] += 1
        if r.weather_condition:
            weathers[r.weather_condition] += 1
    return GenreStats(
        record_count=count,
        top_artists=artists.most_common(TOP_N),
        top_moods=mood_tags.most_common(TOP_N),
        top_places=places.most_common(TOP_N),
        top_weathers=weathers.most_common(3),
    )
