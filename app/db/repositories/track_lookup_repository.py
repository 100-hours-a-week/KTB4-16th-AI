"""track_lookups 접근 — LLM 추천곡의 Spotify 확인 결과 캐시."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TrackLookup


@dataclass(frozen=True)
class SpotifyTrack:
    external_track_id: str
    title: str
    artist_name: str
    spotify_uri: str
    album_image_url: str
    external_url: str


@dataclass(frozen=True)
class LookupEntry:
    lookup_key: str
    llm_artist: str
    llm_title: str
    genre: str
    moods: tuple[str, ...]
    track: SpotifyTrack | None  # None = Spotify에 없는 곡
    verified_at: datetime


class TrackLookupStore(Protocol):
    async def get_many(self, keys: list[str]) -> dict[str, LookupEntry]: ...

    async def save_many(self, entries: list[LookupEntry]) -> None: ...


class TrackLookupRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_many(self, keys: list[str]) -> dict[str, LookupEntry]:
        if not keys:
            return {}
        rows = (
            await self._session.execute(select(TrackLookup).where(TrackLookup.lookup_key.in_(keys)))
        ).scalars()
        return {row.lookup_key: _to_entry(row) for row in rows}

    async def save_many(self, entries: list[LookupEntry]) -> None:
        if not entries:
            return
        values = [_to_values(e) for e in entries]
        stmt = insert(TrackLookup).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[TrackLookup.lookup_key],
            set_={col: stmt.excluded[col] for col in values[0] if col != "lookup_key"},
        )
        await self._session.execute(stmt)
        await self._session.commit()


def _to_entry(row: TrackLookup) -> LookupEntry:
    track = None
    if row.external_track_id:
        track = SpotifyTrack(
            external_track_id=row.external_track_id,
            title=row.title or "",
            artist_name=row.artist_name or "",
            spotify_uri=row.spotify_uri or f"spotify:track:{row.external_track_id}",
            album_image_url=row.album_image_url or "",
            external_url=row.external_url
            or f"https://open.spotify.com/track/{row.external_track_id}",
        )
    return LookupEntry(
        lookup_key=row.lookup_key,
        llm_artist=row.llm_artist,
        llm_title=row.llm_title,
        genre=row.genre,
        moods=tuple(row.moods),
        track=track,
        verified_at=row.verified_at,
    )


def _to_values(e: LookupEntry) -> dict:
    t = e.track
    return {
        "lookup_key": e.lookup_key,
        "llm_artist": e.llm_artist[:200],
        "llm_title": e.llm_title[:300],
        "genre": e.genre,
        "moods": list(e.moods),
        "external_track_id": t.external_track_id if t else None,
        "title": t.title[:300] if t else None,
        "artist_name": t.artist_name[:300] if t else None,
        "spotify_uri": t.spotify_uri if t else None,
        "album_image_url": t.album_image_url if t else None,
        "external_url": t.external_url if t else None,
        "verified_at": e.verified_at,
    }
