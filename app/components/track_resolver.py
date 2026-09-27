"""LLM이 지목한 곡 → 실제 Spotify 트랙 확정. 기능1·3이 공유한다.

곡 확인은 track:/artist: 필드 필터로 한다(SongCandidate.search_query). 그냥 "제목 가수"로
검색하면 LLM이 지어낸 곡이어도 Spotify가 비슷한 아무 곡을 1위로 돌려줘서 걸러지지 않았다.

필드 필터는 띄어쓰기가 다르면 0건을 준다(실측: LLM "아닐 거야" vs Spotify "아닐거야" —
실제로 있는 백예린 곡이 탈락). 그래서 0건이면 일반 검색을 한 번 더 하되, 띄어쓰기·대소문자·
괄호 부가표기를 빼고 비교했을 때 제목과 가수가 둘 다 같은 곡만 받는다.
"""

import asyncio
import logging
import re
from typing import Any

from app.clients.spotify_client import SpotifySearchClient
from app.components.reranker import TrackCandidate
from app.components.song_curator import SongCandidate
from app.exceptions import UpstreamError

logger = logging.getLogger("muro.track_resolver")

# 필드 필터도 제목에 그 단어가 "들어간" 라이브 메들리 같은 버전을 1위로 줄 때가 있어서
# (실측: "Saturday Nite" → "Serpentine Fire / Saturday Nite / ... - Live") 몇 개 받아
# 제목이 똑같은 버전을 우선 고른다. 일반 검색 보정도 노래방·커버가 섞여 같은 개수를 본다.
SEARCH_LIMIT = 5


class TrackResolver:
    def __init__(self, spotify_search: SpotifySearchClient):
        self._spotify = spotify_search

    async def resolve(self, songs: list[SongCandidate]) -> list[TrackCandidate]:
        """추천받은 곡을 실제 Spotify 트랙으로 확정한다.

        곡 하나가 없거나(존재 안 함) 그 곡 조회만 개별 오류가 나도 나머지로 계속
        진행한다 — 8곡 중 1곡 장애로 전체 요청이 실패하던 문제를 여기서 막는다.
        단, 시도한 곡이 전부 오류로 실패하면(=Spotify 자체가 죽은 상태로 추정)
        "0건"이 아니라 실제 장애로 보고 위로 던져서 폴백(degraded)으로 처리한다.

        8곡을 순차로 조회하면 응답이 8~13초까지 걸렸다(실측) — 서로 독립적인
        조회라 병렬로 동시에 보낸다.
        """
        results = await asyncio.gather(
            *(self._find(song) for song in songs), return_exceptions=True
        )

        resolved: list[TrackCandidate] = []
        seen: set[str] = set()
        any_succeeded = False
        last_error: UpstreamError | None = None
        for song, result in zip(songs, results, strict=True):
            if isinstance(result, UpstreamError):
                logger.warning(
                    "Spotify 확인 실패, 이 곡만 건너뜀: %s (%s)", song.search_query, result
                )
                last_error = result
                continue
            if isinstance(result, BaseException):
                raise result
            any_succeeded = True
            if result is None:
                logger.info("추천곡이 Spotify에 없어 제외: %s", song.search_query)
                continue
            track = _to_candidate(result, song)
            if track.external_track_id in seen:
                continue
            seen.add(track.external_track_id)
            resolved.append(track)

        if not any_succeeded and last_error is not None:
            raise last_error
        return resolved

    async def _find(self, song: SongCandidate) -> dict[str, Any] | None:
        items = await self._spotify.search_tracks(song.search_query, limit=SEARCH_LIMIT)
        if items:
            exact = next((i for i in items if _same_title(i["name"], song.title)), None)
            # 제목이 똑같은 게 없으면 필드 필터 1위를 믿는다 — "크러쉬"/"Crush"처럼 표기가
            # 달라도 Spotify가 같은 곡으로 맞춰준 경우라서.
            return exact or items[0]

        items = await self._spotify.search_tracks(f"{song.title} {song.artist}", limit=SEARCH_LIMIT)
        for item in items:
            if _same_title(item["name"], song.title) and _same_artist(item, song.artist):
                return item
        return None


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


def _to_candidate(track_json: dict, song: SongCandidate) -> TrackCandidate:
    images = track_json.get("album", {}).get("images") or []
    return TrackCandidate(
        external_track_id=track_json["id"],
        title=track_json["name"],
        artist_name=", ".join(a["name"] for a in track_json["artists"]),
        spotify_uri=track_json["uri"],
        genre=song.genre,
        moods=song.moods,
        album_image_url=images[0]["url"] if images else None,
        external_url=track_json.get("external_urls", {}).get("spotify")
        or f"https://open.spotify.com/track/{track_json['id']}",
        popularity=track_json.get("popularity", 0),
    )
