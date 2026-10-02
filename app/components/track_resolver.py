"""LLM이 지목한 곡 → 실제 Spotify 트랙 확정. 기능1·3이 공유한다.

곡 확인은 track:/artist: 필드 필터로 한다(SongCandidate.search_query). 그냥 "제목 가수"로
검색하면 LLM이 지어낸 곡이어도 Spotify가 비슷한 아무 곡을 1위로 돌려줘서 걸러지지 않았다.

필드 필터는 띄어쓰기가 다르면 0건을 준다(실측: LLM "아닐 거야" vs Spotify "아닐거야" —
실제로 있는 백예린 곡이 탈락). 그래서 0건이면 일반 검색을 한 번 더 하되, 띄어쓰기·대소문자·
괄호 부가표기를 빼고 비교했을 때 제목과 가수가 둘 다 같은 곡만 받는다.

앨범 커버가 없는 버전은 없는 곡으로 친다. 백엔드 music_tracks.album_image_url이 NOT NULL이라
null을 넘기면 저장이 깨지고, 커버 없는 곡은 대개 정식 발매본이 아니라서 빼도 잃는 게 거의 없다.

확인 결과는 track_lookups에 저장해 두고 유효기간 안에는 Spotify를 다시 부르지 않는다.
추천마다 8~10곡을 조회하다 하루 요청 한도를 넘겨 추천이 전부 멈춘 적이 있어서다
(2026-10-02). 유효기간은 Spotify 약관(메타데이터 임시 캐시만 허용, 최신 상태 유지)에 맞춰
찾은 곡 30일, 없는 곡 1일로 둔다. 없는 곡을 짧게 두는 건 발매 직후라 아직 검색이 안 되는
경우를 다시 확인하기 위해서다.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from app.clients.spotify_client import SpotifySearchClient
from app.components.reranker import TrackCandidate
from app.components.song_curator import SongCandidate
from app.db.repositories.track_lookup_repository import LookupEntry, SpotifyTrack, TrackLookupStore
from app.exceptions import UpstreamError

logger = logging.getLogger("muro.track_resolver")

# 필드 필터도 제목에 그 단어가 "들어간" 라이브 메들리 같은 버전을 1위로 줄 때가 있어서
# (실측: "Saturday Nite" → "Serpentine Fire / Saturday Nite / ... - Live") 몇 개 받아
# 제목이 똑같은 버전을 우선 고른다. 일반 검색 보정도 노래방·커버가 섞여 같은 개수를 본다.
SEARCH_LIMIT = 5

FOUND_TTL = timedelta(days=30)
NOT_FOUND_TTL = timedelta(days=1)


class TrackResolver:
    def __init__(
        self,
        spotify_search: SpotifySearchClient,
        store: TrackLookupStore | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._spotify = spotify_search
        self._store = store
        self._now = now

    async def resolve(self, songs: list[SongCandidate]) -> list[TrackCandidate]:
        """추천받은 곡을 실제 Spotify 트랙으로 확정한다.

        곡 하나가 없거나(존재 안 함) 그 곡 조회만 개별 오류가 나도 나머지로 계속
        진행한다 — 8곡 중 1곡 장애로 전체 요청이 실패하던 문제를 여기서 막는다.
        단, 시도한 곡이 전부 오류로 실패하면(=Spotify 자체가 죽은 상태로 추정)
        "0건"이 아니라 실제 장애로 보고 위로 던져서 폴백(degraded)으로 처리한다.

        8곡을 순차로 조회하면 응답이 8~13초까지 걸렸다(실측) — 서로 독립적인
        조회라 병렬로 동시에 보낸다.
        """
        cached = await self._load_cached(songs)
        # 같은 곡을 LLM이 두 번 적은 경우 한 번만 조회한다 — 한 번에 같은 키를 두 번 저장하면
        # Postgres ON CONFLICT가 실패해 캐시 저장이 통째로 빠진다.
        to_search = list(
            {_lookup_key(s): s for s in songs if _lookup_key(s) not in cached}.values()
        )
        results = await asyncio.gather(
            *(self._find(song) for song in to_search), return_exceptions=True
        )

        found: dict[str, SpotifyTrack | None] = {key: entry.track for key, entry in cached.items()}
        fresh: list[LookupEntry] = []
        any_succeeded = bool(cached)
        last_error: UpstreamError | None = None
        for song, result in zip(to_search, results, strict=True):
            if isinstance(result, UpstreamError):
                logger.warning(
                    "Spotify 확인 실패, 이 곡만 건너뜀: %s (%s)", song.search_query, result
                )
                last_error = result
                continue
            if isinstance(result, BaseException):
                raise result
            any_succeeded = True
            track = _to_spotify_track(result) if result is not None else None
            found[_lookup_key(song)] = track
            fresh.append(_to_entry(song, track, self._now()))

        if not any_succeeded and last_error is not None:
            raise last_error
        await self._save(fresh)

        resolved: list[TrackCandidate] = []
        seen: set[str] = set()
        for song in songs:
            track = found.get(_lookup_key(song))
            if track is None:
                if _lookup_key(song) in found:
                    logger.info("추천곡이 Spotify에 없어 제외: %s", song.search_query)
                continue
            if track.external_track_id in seen:
                continue
            seen.add(track.external_track_id)
            resolved.append(_to_candidate(track, song))
        return resolved

    async def _load_cached(self, songs: list[SongCandidate]) -> dict[str, LookupEntry]:
        """유효기간 안의 저장된 확인 결과. 캐시 장애로 추천이 멈추면 안 되니 실패하면 빈 값."""
        if self._store is None:
            return {}
        try:
            entries = await self._store.get_many(list({_lookup_key(s) for s in songs}))
        except Exception:
            logger.exception("곡 확인 캐시 조회 실패 — Spotify로 직접 확인")
            return {}
        now = self._now()
        return {
            key: entry
            for key, entry in entries.items()
            if now - entry.verified_at < (FOUND_TTL if entry.track else NOT_FOUND_TTL)
        }

    async def _save(self, entries: list[LookupEntry]) -> None:
        if self._store is None or not entries:
            return
        try:
            await self._store.save_many(entries)
        except Exception:
            logger.exception("곡 확인 캐시 저장 실패 — 응답은 그대로 진행")

    async def _find(self, song: SongCandidate) -> dict[str, Any] | None:
        items = _with_album_image(
            await self._spotify.search_tracks(song.search_query, limit=SEARCH_LIMIT)
        )
        if items:
            exact = next((i for i in items if _same_title(i["name"], song.title)), None)
            # 제목이 똑같은 게 없으면 필드 필터 1위를 믿는다 — "크러쉬"/"Crush"처럼 표기가
            # 달라도 Spotify가 같은 곡으로 맞춰준 경우라서.
            return exact or items[0]

        items = _with_album_image(
            await self._spotify.search_tracks(f"{song.title} {song.artist}", limit=SEARCH_LIMIT)
        )
        for item in items:
            if _same_title(item["name"], song.title) and _same_artist(item, song.artist):
                return item
        return None


def _with_album_image(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if (i.get("album") or {}).get("images")]


def _normalize(text: str) -> str:
    """비교용 — 괄호 부가표기·" - Remastered" 꼬리·띄어쓰기·문장부호·대소문자를 뺀다."""
    text = re.sub(r"[(\[].*?[)\]]", "", text)
    text = text.split(" - ")[0]
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def _same_title(spotify_title: str, llm_title: str) -> bool:
    return bool(_normalize(llm_title)) and _normalize(spotify_title) == _normalize(llm_title)


def _same_artist(track_json: dict[str, Any], llm_artist: str) -> bool:
    wanted = _normalize(llm_artist)
    return bool(wanted) and any(_normalize(a["name"]) == wanted for a in track_json["artists"])


def _lookup_key(song: SongCandidate) -> str:
    """같은 곡을 LLM이 조금 다르게 적어도(띄어쓰기·대소문자·괄호) 같은 키가 되게."""
    return f"{_normalize(song.artist)}|{_normalize(song.title)}"[:300]


def _to_spotify_track(track_json: dict[str, Any]) -> SpotifyTrack:
    return SpotifyTrack(
        external_track_id=track_json["id"],
        title=track_json["name"],
        artist_name=", ".join(a["name"] for a in track_json["artists"]),
        spotify_uri=track_json["uri"],
        album_image_url=track_json["album"]["images"][0]["url"],
        external_url=track_json.get("external_urls", {}).get("spotify")
        or f"https://open.spotify.com/track/{track_json['id']}",
    )


def _to_entry(song: SongCandidate, track: SpotifyTrack | None, now: datetime) -> LookupEntry:
    return LookupEntry(
        lookup_key=_lookup_key(song),
        llm_artist=song.artist,
        llm_title=song.title,
        genre=song.genre,
        moods=song.moods,
        track=track,
        verified_at=now,
    )


def _to_candidate(track: SpotifyTrack, song: SongCandidate) -> TrackCandidate:
    # 장르·무드는 저장값이 아니라 이번 추천에서 LLM이 붙인 값을 쓴다 — 재랭킹은 이번
    # 상황 태그와 같은 호출에서 나온 곡 태그끼리 비교해야 기준이 맞는다.
    return TrackCandidate(
        external_track_id=track.external_track_id,
        title=track.title,
        artist_name=track.artist_name,
        spotify_uri=track.spotify_uri,
        genre=song.genre,
        moods=song.moods,
        album_image_url=track.album_image_url,
        external_url=track.external_url,
    )
