"""기능3 처리 순서: CLIP 태깅 → LLM 리라이팅 → Spotify 검색(장르로 넓게)
→ 재랭킹(무드로 좁게) → 상위 3곡.

에러 처리 원칙(위키 5-1절): CLIP·LLM 장애는 폴백 + degraded:true, 검색 0건은
에러가 아니라 그냥 빈 배열(degraded:false). 재랭킹 실패는 검색 결과 자체는
유효하므로 순서만 원본으로 두고 응답은 그대로 낸다(요청을 실패시키지 않음).

※ "지역 인기곡 폴백"은 백엔드에 해당 조회 API(FR-008 집계)가 아직 없어
  place만 받아두고 실제 조회는 미구현 — 지금은 빈 tracks로 대체한다.
"""

import logging

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.query_rewriter import QueryRewriter
from app.components.reranker import Reranker, TrackCandidate
from app.exceptions import ClipError, LLMError, UpstreamError
from app.schemas.common import Track
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse

logger = logging.getLogger("muro.photo_recommend")

# Spotify Search API 실측: limit이 10을 넘으면 400 Invalid limit (Development Mode 제한으로 추정)
SEARCH_LIMIT = 10
RESULT_COUNT = 3


class PhotoRecommendService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        query_rewriter: QueryRewriter,
        reranker: Reranker,
        spotify_search: SpotifySearchClient,
    ):
        self._clip_tagger = clip_tagger
        self._query_rewriter = query_rewriter
        self._reranker = reranker
        self._spotify = spotify_search

    async def recommend(self, req: PhotoRecommendRequest) -> PhotoRecommendResponse:
        try:
            tags = await self._clip_tagger.tag(req.image_url)
        except ClipError as e:
            logger.warning("CLIP 태깅 실패, 폴백: %s", e)
            return self._fallback()

        if not tags:
            # 임계치 넘는 태그가 하나도 없음 — 검색어를 만들 근거가 없어 폴백과 동일 처리
            return self._fallback()

        try:
            rewritten_query = await self._query_rewriter.rewrite_from_tags(tags)
        except LLMError as e:
            logger.warning("검색어 리라이팅 실패, 폴백: %s", e)
            return self._fallback(tags=tags)

        retry_count = 0
        try:
            candidates = await self._search(rewritten_query)
            if not candidates:
                relaxed_query = _last_word(rewritten_query)
                if relaxed_query != rewritten_query:
                    retry_count = 1
                    candidates = await self._search(relaxed_query)
        except UpstreamError as e:
            logger.warning("Spotify 검색 실패, 폴백: %s", e)
            return self._fallback(tags=tags, rewritten_query=rewritten_query)

        if not candidates:
            # 검색 0건은 에러가 아니다 — 그대로 빈 결과 반환 (재시도는 이미 해봤을 수 있음)
            return PhotoRecommendResponse(
                mood_tags=tags,
                rewritten_query=rewritten_query,
                tracks=[],
                degraded=False,
                retry_count=retry_count,
            )

        ranked = await self._rerank_safely(rewritten_query, tags, candidates)
        top = ranked[:RESULT_COUNT]
        return PhotoRecommendResponse(
            mood_tags=tags,
            rewritten_query=rewritten_query,
            tracks=[_to_track(c) for c in top],
            degraded=False,
            retry_count=retry_count,
        )

    async def _search(self, query: str) -> list[TrackCandidate]:
        items = await self._spotify.search_tracks(query, limit=SEARCH_LIMIT)
        return [_to_candidate(item) for item in items]

    async def _rerank_safely(
        self, query: str, tags: list[str], candidates: list[TrackCandidate]
    ) -> list[TrackCandidate]:
        try:
            return await self._reranker.rerank(", ".join(tags), candidates)
        except Exception:
            logger.exception("재랭킹 실패 — 검색 결과 순서 그대로 사용")
            return candidates

    def _fallback(
        self, tags: list[str] | None = None, rewritten_query: str = ""
    ) -> PhotoRecommendResponse:
        return PhotoRecommendResponse(
            mood_tags=tags or [], rewritten_query=rewritten_query, tracks=[], degraded=True
        )


def _last_word(query: str) -> str:
    """0건이면 검색어를 느슨하게 — 보통 마지막 단어가 장르 명사라 그것만 남긴다."""
    parts = query.split()
    return parts[-1] if parts else query


def _to_candidate(track_json: dict) -> TrackCandidate:
    images = track_json.get("album", {}).get("images") or []
    return TrackCandidate(
        external_track_id=track_json["id"],
        title=track_json["name"],
        artist_name=", ".join(a["name"] for a in track_json["artists"]),
        spotify_uri=track_json["uri"],
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
