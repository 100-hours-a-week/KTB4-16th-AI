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

2026-10-16: 기능3처럼 최대 5곡 중 일부(기본 3곡)를 곡 풀(song_pool.py)에서 채운다. LLM이
한국 곡을 잘 몰라서 실존하는 곡 풀 비중을 절반 이상으로 둔다. 곡 풀은 LLM이 준 상황
장르·무드로 고르고, 개인 취향(과거 자물쇠 곡)은 LLM 몫에만 반영된다. LLM에게는 그만큼
적게(기본 4곡) 받는다. 곡 풀 곡도 같은 점수 기준(MIN_RERANK_SCORE)을 넘어야 남는다.

재추천: 클라이언트가 excludeTrackIds로 직전까지 보여준 곡을 누적해서 보내면 그 곡들을
빼고 추천한다(개인 취향 참고곡 제외와 같은 방식으로 합쳐서 처리). 서버는 상태를 들고
있지 않는다.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.reranker import Reranker, TrackCandidate
from app.components.song_curator import (
    CANDIDATE_COUNT,
    CurationResult,
    SongCandidate,
    SongCurator,
)
from app.components.song_mix import resolve_both
from app.components.song_pool import SongPool
from app.components.track_resolver import TrackResolver
from app.components.vector_search import RecordSearch
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.exceptions import ExternalApiFailedError, LLMError, UpstreamError
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
# 2026-10-02 팀 결정: 추천은 최대 5곡. 백엔드가 limit을 더 크게 보내도(당시 10) 5곡까지만
# 주고, 5곡이 찼으면 LLM 보충 호출도 하지 않는다.
MAX_TRACKS = 5

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
        track_lookups: TrackLookupStore | None = None,
        song_pool: SongPool | None = None,
        pool_count: int = 0,
        llm_count: int = CANDIDATE_COUNT,
    ):
        self._pool = song_pool if song_pool is not None and pool_count > 0 else None
        self._pool_count = pool_count if self._pool is not None else 0
        # 곡 풀을 안 쓰면 예전처럼 LLM에게 넉넉히 받는다
        self._llm_count = llm_count if self._pool is not None else CANDIDATE_COUNT
        self._embedder = embedder
        self._records = record_search
        self._song_curator = song_curator
        self._reranker = reranker
        self._spotify = spotify_search
        self._resolver = TrackResolver(spotify_search, track_lookups)

    async def recommend(self, req: ContextRecommendRequest) -> ContextRecommendResponse:
        moment = f"{_weather_phrase(req.weather.condition)} {_time_bucket(req.local_time)}"
        references = await self._personal_references(req.user_id, moment)
        basis = _basis(references, req)

        prompt = _situation_prompt(req, moment, references)
        try:
            curation = await self._song_curator.curate(prompt, count=self._llm_count)
        except LLMError as e:
            logger.warning("곡 추천 실패, 폴백: %s", e)
            return _degraded(req, basis)

        pool_songs = self._pick_from_pool(curation, references)
        if not curation.songs and not pool_songs:
            logger.warning("곡 추천 결과를 파싱하지 못함, 폴백")
            return _degraded(req, basis)

        try:
            llm_found, pool_found = await resolve_both(self._resolver, curation.songs, pool_songs)
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패: %s", e)
            raise ExternalApiFailedError("Spotify 곡 확인에 실패했어요.") from e

        excluded = {r.external_track_id for r in references} | set(req.exclude_track_ids)
        llm_found = [c for c in llm_found if c.external_track_id not in excluded]
        pool_found = [c for c in pool_found if c.external_track_id not in excluded]
        limit = min(req.limit, MAX_TRACKS)
        kept = self._mix(curation, llm_found, pool_found, limit)

        if len(kept) < limit:
            # Spotify 탈락·점수 미달로 모자라면 이미 나온 곡을 빼고 LLM에게 한 번만 더 받는다.
            # 보충은 있으면 좋은 것이라 실패해도 지금 있는 곡으로 응답한다.
            try:
                extra = await self._song_curator.curate(
                    prompt, exclude=curation.songs + pool_songs, count=self._llm_count
                )
                more = await self._resolver.resolve(extra.songs)
            except UpstreamError as e:
                logger.warning("곡 보충 실패, 있는 곡으로 응답: %s", e)
            else:
                seen = {c.external_track_id for c in llm_found + pool_found} | excluded
                llm_found += [c for c in more if c.external_track_id not in seen]
                kept = self._mix(curation, llm_found, pool_found, limit)

        return ContextRecommendResponse(
            request_id=req.request_id,
            rewritten_query=_tags_label(curation.situation_genres, curation.situation_moods),
            recommendation_basis=basis,
            tracks=[_to_ranked_track(c, score) for c, score in kept],
        )

    def _keep(
        self, curation: CurationResult, candidates: list[TrackCandidate]
    ) -> list[tuple[TrackCandidate, float]]:
        """점수 순으로 줄 세우고 기준 미만은 뺀다."""
        ranked = self._reranker.rank(
            curation.situation_genres, curation.situation_moods, candidates
        )
        return [(c, score) for c, score in ranked if score >= MIN_RERANK_SCORE]

    def _mix(
        self,
        curation: CurationResult,
        llm_found: list[TrackCandidate],
        pool_found: list[TrackCandidate],
        limit: int,
    ) -> list[tuple[TrackCandidate, float]]:
        """곡 풀 pool_count곡 + 나머지는 LLM 곡. 한쪽이 모자라면 다른 쪽으로 채우고 점수 순으로."""
        pool_kept = self._keep(curation, pool_found)
        llm_kept = self._keep(curation, llm_found)
        pool_first = pool_kept[: min(self._pool_count, limit)]
        ordered = (
            pool_first
            + llm_kept[: limit - len(pool_first)]
            + pool_kept[len(pool_first) :]
            + llm_kept[limit - len(pool_first) :]
        )
        mixed: list[tuple[TrackCandidate, float]] = []
        seen: set[str] = set()
        for c, score in ordered:
            if c.external_track_id in seen:
                continue
            seen.add(c.external_track_id)
            mixed.append((c, score))
            if len(mixed) == limit:
                break
        return sorted(mixed, key=lambda pair: pair[1], reverse=True)

    def _pick_from_pool(
        self, curation: CurationResult, references: list["ReferenceTrack"]
    ) -> list[SongCandidate]:
        if self._pool is None:
            return []
        heard = [SongCandidate(_first_artist(r.artist), r.title, "", ()) for r in references]
        try:
            return self._pool.pick(
                curation.situation_genres,
                curation.situation_moods,
                self._pool_count,
                exclude=curation.songs + heard,
            )
        except Exception:
            logger.exception("곡 풀에서 고르기 실패 — LLM 곡만 사용")
            return []

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


def _first_artist(artists: str) -> str:
    return re.split(r",\s*", artists)[0]


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
