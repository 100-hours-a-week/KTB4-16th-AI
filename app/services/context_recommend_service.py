"""기능1 처리 순서: 상황 문장 만들기 → 비슷한 과거 자물쇠 찾기 → LLM 곡+태그 추천
→ Spotify 실존 확인 → 태그 겹침 재랭킹 → 기준 미만 제외 → 최대 limit곡.

위키 설계는 "LLM 검색어 → Spotify 검색 → 임베딩 유사도 재랭킹"이었는데, 기능3에서 이 방식이
배경음악·컴필레이션 채널 곡만 걸리고 재랭킹 순서도 거꾸로 나오는 걸 실측으로 확인해서
기능3과 같은 구조(LLM이 곡 지목 → track:/artist:로 확인 → 태그 겹침)로 만들었다.

추천 근거(recommendationBasis)는 3단계:
  PERSONAL — 이 사용자의 과거 자물쇠 코멘트 중 지금 상황("비 오는 저녁")과 비슷한 게 있으면
             그때 들은 곡을 LLM에 취향 참고용으로 넘긴다
  REGIONAL — 개인 기록은 없고, 백엔드가 넘긴 주변 자물쇠 인기곡(nearbyTracks)이 있을 때
  GENERIC  — 둘 다 없으면 날씨·시간대만으로
주변 인기곡은 개인 기록이 있어도 프롬프트에 같이 넣는다(근거 표기만 PERSONAL이 우선).
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.reranker import Reranker, TrackCandidate
from app.components.song_curator import SongCurator
from app.components.track_resolver import TrackResolver
from app.components.vector_search import RecordSearch
from app.exceptions import LLMError, UpstreamError
from app.schemas.context_recommend import (
    ContextRecommendRequest,
    ContextRecommendResponse,
    RankedTrack,
)

logger = logging.getLogger("muro.context_recommend")

# 2026-09-26 더미 자물쇠(유저 5명, 코멘트 43개)로 실측: 비슷한 코멘트는 0.40~0.74
# ("노을 지는 저녁"↔"노을이 미쳤다" 0.74, "쌀쌀한 밤"↔"조금 쌀쌀했지만 걷기 좋았어" 0.47),
# 상관없는 코멘트는 0.26~0.39. 더미 코멘트가 짧고 반복이 많아서 대략적인 값 —
# 실제 사용자 코멘트가 쌓이면 다시 맞춰야 한다.
PERSONAL_SIMILARITY_THRESHOLD = 0.40
SIMILAR_SEARCH_LIMIT = 5
REFERENCE_COUNT = 3
# 백엔드가 인기순으로 보내므로 앞에서부터. 너무 많으면 LLM이 인기곡만 따라 뽑는다
NEARBY_REFERENCE_COUNT = 5

# 장르도 무드도 하나도 안 겹치는 곡만 뺀다(상황 무드 3개 중 1개만 겹쳐도 0.117).
# 위키 설계상 임계값은 실측 후 튜닝.
MIN_RERANK_SCORE = 0.1

# 백엔드 WeatherCondition enum(기상청 단기예보 기준) → 상황 문장용 표현. 모르는 값은 그대로 쓴다.
WEATHER_PHRASES = {
    "CLEAR": "맑은",
    "CLOUDY": "구름 많은",
    "OVERCAST": "흐린",
    "RAIN": "비 오는",
    "SNOW": "눈 오는",
    "RAIN_SNOW": "진눈깨비 오는",
    "SHOWER": "소나기 오는",
}


@dataclass(frozen=True)
class ReferenceTrack:
    external_track_id: str
    title: str
    artist: str


class ContextRecommendService:
    def __init__(
        self,
        embedder: Embedder,
        record_search: RecordSearch,
        song_curator: SongCurator,
        reranker: Reranker,
        spotify_search: SpotifySearchClient,
    ):
        self._embedder = embedder
        self._records = record_search
        self._song_curator = song_curator
        self._reranker = reranker
        self._spotify = spotify_search
        self._resolver = TrackResolver(spotify_search)

    async def recommend(self, req: ContextRecommendRequest) -> ContextRecommendResponse:
        moment = f"{_weather_phrase(req.weather.condition)} {_time_bucket(req.local_time)}"
        references = await self._personal_references(req.user_id, moment)
        basis = _basis(references, req)

        try:
            curation = await self._song_curator.curate(_situation_prompt(req, moment, references))
        except LLMError as e:
            logger.warning("곡 추천 실패, 폴백: %s", e)
            return _degraded(req, basis)

        if not curation.songs:
            logger.warning("곡 추천 결과를 파싱하지 못함, 폴백")
            return _degraded(req, basis)

        try:
            candidates = await self._resolver.resolve(curation.songs)
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패, 폴백: %s", e)
            return _degraded(req, basis)

        already_heard = {r.external_track_id for r in references}
        candidates = [c for c in candidates if c.external_track_id not in already_heard]

        ranked = self._reranker.rank(
            curation.situation_genres, curation.situation_moods, candidates
        )
        kept = [(c, score) for c, score in ranked if score >= MIN_RERANK_SCORE][: req.limit]

        return ContextRecommendResponse(
            request_id=req.request_id,
            rewritten_query=_tags_label(curation.situation_genres, curation.situation_moods),
            recommendation_basis=basis,
            tracks=[_to_ranked_track(c, score) for c, score in kept],
        )

    async def _personal_references(self, user_id: int, moment: str) -> list[ReferenceTrack]:
        """지금 상황과 비슷한 과거 자물쇠에서 들은 곡. 못 찾거나 실패하면 빈 목록(=GENERIC).

        개인화는 있으면 좋은 부가 신호라, 여기서 난 오류로 추천 전체를 실패시키지 않는다.
        """
        try:
            query_vector = await self._embedder.embed(moment)
            hits = await self._records.similar_by_comment(
                user_id, query_vector, SIMILAR_SEARCH_LIMIT
            )
        except Exception:
            logger.exception("개인 기록 검색 실패 — GENERIC으로 진행")
            return []

        track_ids: list[str] = []
        for hit in hits:
            if hit.similarity < PERSONAL_SIMILARITY_THRESHOLD or not hit.external_track_id:
                continue
            if hit.external_track_id not in track_ids:
                track_ids.append(hit.external_track_id)
        track_ids = track_ids[:REFERENCE_COUNT]
        if not track_ids:
            return []

        # 과거 자물쇠엔 곡 ID만 있고 제목·가수는 저장돼 있지 않아서 Spotify에서 가져온다.
        results = await asyncio.gather(
            *(self._spotify.get_track(track_id) for track_id in track_ids),
            return_exceptions=True,
        )
        references: list[ReferenceTrack] = []
        for track_id, result in zip(track_ids, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("참고곡 조회 실패, 건너뜀: %s (%s)", track_id, result)
                continue
            references.append(
                ReferenceTrack(
                    external_track_id=track_id,
                    title=result["name"],
                    artist=", ".join(a["name"] for a in result["artists"]),
                )
            )
        return references


def _basis(references: list[ReferenceTrack], req: ContextRecommendRequest) -> str:
    if references:
        return "PERSONAL"
    if req.nearby_tracks:
        return "REGIONAL"
    return "GENERIC"


def _weather_phrase(condition: str) -> str:
    return WEATHER_PHRASES.get(condition.strip().upper(), condition.strip())


def _time_bucket(local_time: datetime) -> str:
    """요청에 담긴 오프셋 기준 시각으로 시간대를 나눈다 (19:40 → 저녁)."""
    hour = local_time.hour
    if hour < 5:
        return "새벽"
    if hour < 11:
        return "아침"
    if hour < 17:
        return "낮"
    if hour < 21:
        return "저녁"
    return "밤"


def _situation_prompt(
    req: ContextRecommendRequest, moment: str, references: list[ReferenceTrack]
) -> str:
    parts = [moment]
    if req.weather.temperature is not None:
        parts.append(f"기온 {req.weather.temperature:g}도")
    if req.place and req.place.name:
        parts.append(f"장소: {req.place.name}")
    prompt = f"상황: {', '.join(parts)}"
    if references:
        heard = ", ".join(f"{r.artist} - {r.title}" for r in references)
        prompt += f"\n이 사용자가 비슷한 상황에서 들었던 곡(취향 참고용): {heard}"
    if req.nearby_tracks:
        nearby = ", ".join(
            f"{t.artist_name} - {t.title}" for t in req.nearby_tracks[:NEARBY_REFERENCE_COUNT]
        )
        prompt += f"\n이 근처 사람들이 최근 자물쇠에 많이 단 곡(분위기 참고용): {nearby}"
    return prompt


def _tags_label(genres: tuple[str, ...], moods: tuple[str, ...]) -> str:
    return " · ".join(part for part in (", ".join(genres), ", ".join(moods)) if part)


def _to_ranked_track(c: TrackCandidate, score: float) -> RankedTrack:
    return RankedTrack(
        title=c.title,
        artist_name=c.artist_name,
        external_track_id=c.external_track_id,
        spotify_uri=c.spotify_uri,
        album_image_url=c.album_image_url,
        external_url=c.external_url,
        rerank_score=round(score, 3),
    )


def _degraded(req: ContextRecommendRequest, basis: str) -> ContextRecommendResponse:
    return ContextRecommendResponse(
        request_id=req.request_id,
        rewritten_query="",
        recommendation_basis=basis,
        tracks=[],
        degraded=True,
    )
