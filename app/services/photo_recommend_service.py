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

import logging

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.reranker import Reranker, TrackCandidate
from app.components.song_curator import CurationResult, SongCurator
from app.components.track_resolver import TrackResolver
from app.exceptions import ClipError, LLMError, UpstreamError
from app.schemas.common import Track
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse

logger = logging.getLogger("muro.photo_recommend")

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
        self._resolver = TrackResolver(spotify_search)

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
            candidates = await self._resolver.resolve(curation.songs)
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

    def _rerank_safely(
        self, curation: CurationResult, candidates: list[TrackCandidate]
    ) -> list[TrackCandidate]:
        try:
            return self._reranker.rerank(
                curation.situation_genres, curation.situation_moods, candidates
            )
        except Exception:
            logger.exception("재랭킹 실패 — LLM이 준 순서 그대로 사용")
            return candidates

    def _fallback(self, tags: list[str] | None = None) -> PhotoRecommendResponse:
        return PhotoRecommendResponse(
            mood_tags=tags or [], rewritten_query="", tracks=[], degraded=True
        )


def _to_track(c: TrackCandidate) -> Track:
    return Track(
        title=c.title,
        artist_name=c.artist_name,
        external_track_id=c.external_track_id,
        spotify_uri=c.spotify_uri,
        album_image_url=c.album_image_url,
        external_url=c.external_url,
    )
