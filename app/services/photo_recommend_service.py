"""기능3 처리 순서: CLIP 태깅 → LLM 곡+태그 추천 → Spotify 실존 확인 → 재랭킹 → 상위 3곡.

이전엔 LLM이 만든 장르 키워드로 Spotify를 넓게 검색하고 그 결과를 재랭킹으로 걸렀다.
그다음엔 LLM이 실제 곡을 직접 추천하고 임베딩 유사도로 재랭킹했는데, 자유 문장
임베딩이 "신나는지 잔잔한지" 같은 축을 신뢰성 있게 구분 못 했다(자세한 배경은
song_curator.py, reranker.py). 그래서 지금은 song_curator가 곡 추천과 동시에
사진 상황·각 곡의 장르·무드 태그까지 같은 호출에서 받고, 재랭킹은 그 태그가
얼마나 겹치는지로 점수를 매긴다.

에러 처리 원칙(위키 5-1절): CLIP·LLM 장애는 폴백 + degraded:true, 결과 0건은 에러가
아니라 빈 배열(degraded:false). 재랭킹 실패는 후보 자체는 유효하므로 순서만 원본으로
두고 응답은 그대로 낸다.

※ "지역 인기곡 폴백"은 백엔드에 해당 조회 API(FR-008 집계)가 아직 없어
  place만 받아두고 실제 조회는 미구현 — 지금은 빈 tracks로 대체한다.
"""

import asyncio
import logging

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.reranker import Reranker, TrackCandidate
from app.components.song_curator import CurationResult, SongCandidate, SongCurator
from app.exceptions import ClipError, LLMError, UpstreamError
from app.schemas.common import Track
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse

logger = logging.getLogger("muro.photo_recommend")

# 제목+가수로 지목해서 찾는 정밀 검색이라 1위만 본다. 여기서 안 걸리면 없는 곡으로 친다.
RESOLVE_LIMIT = 1
RESULT_COUNT = 3


class PhotoRecommendService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        song_curator: SongCurator,
        reranker: Reranker,
        spotify_search: SpotifySearchClient,
    ):
        self._clip_tagger = clip_tagger
        self._song_curator = song_curator
        self._reranker = reranker
        self._spotify = spotify_search

    async def recommend(self, req: PhotoRecommendRequest) -> PhotoRecommendResponse:
        try:
            tags = await self._clip_tagger.tag(req.image_url)
        except ClipError as e:
            logger.warning("CLIP 태깅 실패, 폴백: %s", e)
            return self._fallback()

        if not tags:
            # 임계치 넘는 태그가 하나도 없음 — 곡을 고를 근거가 없어 폴백과 동일 처리
            return self._fallback()

        try:
            curation = await self._song_curator.curate_from_tags(tags)
        except LLMError as e:
            logger.warning("곡 추천 실패, 폴백: %s", e)
            return self._fallback(tags=tags)

        if not curation.songs:
            # 형식이 깨져 한 줄도 못 건졌다 — 추천 자체가 실패한 것으로 본다
            logger.warning("곡 추천 결과를 파싱하지 못함, 폴백")
            return self._fallback(tags=tags)

        try:
            candidates = await self._resolve_on_spotify(curation.songs)
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패, 폴백: %s", e)
            return self._fallback(tags=tags)

        if not candidates:
            # 추천받은 곡이 전부 Spotify에 없었다 — 에러는 아니고 빈 결과
            return PhotoRecommendResponse(mood_tags=tags, rewritten_query="", tracks=[])

        ranked = self._rerank_safely(curation, candidates)
        return PhotoRecommendResponse(
            mood_tags=tags,
            rewritten_query="",
            tracks=[_to_track(c) for c in ranked[:RESULT_COUNT]],
        )

    async def _resolve_on_spotify(self, songs: list[SongCandidate]) -> list[TrackCandidate]:
        """추천받은 곡을 실제 Spotify 트랙으로 확정한다.

        곡 하나가 없거나(존재 안 함) 그 곡 조회만 개별 오류가 나도 나머지로 계속
        진행한다 — 8곡 중 1곡 장애로 전체 요청이 실패하던 문제를 여기서 막는다.
        단, 시도한 곡이 전부 오류로 실패하면(=Spotify 자체가 죽은 상태로 추정)
        "0건"이 아니라 실제 장애로 보고 위로 던져서 폴백(degraded)으로 처리한다.

        8곡을 순차로 조회하면 응답이 8~13초까지 걸렸다(실측) — 서로 독립적인
        조회라 병렬로 동시에 보낸다.
        """
        results = await asyncio.gather(
            *(
                self._spotify.search_tracks(song.search_query, limit=RESOLVE_LIMIT)
                for song in songs
            ),
            return_exceptions=True,
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
            if not result:
                logger.info("추천곡이 Spotify에 없어 제외: %s", song.search_query)
                continue
            track = _to_candidate(result[0], song)
            if track.external_track_id in seen:
                continue
            seen.add(track.external_track_id)
            resolved.append(track)

        if not any_succeeded and last_error is not None:
            raise last_error
        return resolved

    def _rerank_safely(
        self, curation: CurationResult, candidates: list[TrackCandidate]
    ) -> list[TrackCandidate]:
        try:
            return self._reranker.rerank(
                curation.situation_genre, curation.situation_moods, candidates
            )
        except Exception:
            logger.exception("재랭킹 실패 — LLM이 준 순서 그대로 사용")
            return candidates

    def _fallback(self, tags: list[str] | None = None) -> PhotoRecommendResponse:
        return PhotoRecommendResponse(
            mood_tags=tags or [], rewritten_query="", tracks=[], degraded=True
        )


def _to_candidate(track_json: dict, song: SongCandidate) -> TrackCandidate:
    images = track_json.get("album", {}).get("images") or []
    return TrackCandidate(
        external_track_id=track_json["id"],
        title=track_json["name"],
        artist_name=", ".join(a["name"] for a in track_json["artists"]),
        spotify_uri=track_json["uri"],
        genre=song.genre,
        moods=song.moods,
        album_art_url=images[0]["url"] if images else None,
        popularity=track_json.get("popularity", 0),
    )


def _to_track(c: TrackCandidate) -> Track:
    return Track(
        title=c.title,
        artist=c.artist_name,
        spotify_uri=c.spotify_uri,
        album_art_url=c.album_art_url,
    )
