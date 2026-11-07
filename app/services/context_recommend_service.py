"""기능1 처리 순서: 상황 문장 만들기 → 장면 3개·문장 2종(Gemma) → 곡풀 벡터 점수 → 취향 장르 거르기
→ 상위 곡 중 무작위 → Spotify 실존 확인 → 최대 limit곡.

2026-11: 상용 LLM(Claude)을 빼고 RunPod의 Gemma로 바꿨다. 예전에는 Claude가 상황의 장르·무드·분위기
문장을 쓰고 곡풀을 장르·무드 태그로 거른 뒤 벡터로 줄 세웠다. 지금은 Gemma가 상황(계절·요일·시간대·
날씨·기온)에서 장면 3개를 상상하고 장면마다 가사 문체 문장과 설명 문체 문장을 쓴다. 곡풀의 곡마다
가사 벡터·곡 설명 벡터가 있어서(song_pool.py) 각 벡터와 같은 문체의 문장을 비교해 점수를 내고
(장면 3개 평균), 사용자 취향 장르(온보딩 + 자물쇠 곡 장르)로 후보를 거른 뒤 상위 곡에서 무작위로
5곡을 뽑는다 (scene_picker.py). 한 자리는 취향 밖 장르 곡이다.

추천 근거(recommendationBasis)는 3단계:
  PERSONAL — 이 사용자의 과거 자물쇠 코멘트 중 지금 상황("비 오는 저녁")과 비슷한 게 있으면
             그때 들은 곡은 이번 추천에서 뺀다
  REGIONAL — 개인 기록은 없고, 백엔드가 넘긴 주변 자물쇠 인기곡(nearbyTracks)이 있을 때
  GENERIC  — 둘 다 없으면 날씨·시간대만으로
주변 인기곡은 곡풀에 있는 곡의 가사 벡터를 질문 가사 벡터 쪽으로 조금 섞는 데 쓴다(nearby_weight).

재추천: 클라이언트가 excludeTrackIds로 직전까지 보여준 곡을 누적해서 보내면 그 곡들을
빼고 추천한다. 곡풀 곡은 Spotify ID를 확인한 뒤에야 알 수 있어서 그만큼 더 뽑아 둔다. 서버는
상태를 들고 있지 않는다.

응답의 recommendReason은 상황 값과 고른 곡의 장르로 만든 한 줄이다 (recommend_reason.py).
장면 문장은 로그에만 남긴다 — 그대로 보여 줬더니 엉뚱한 문장이 나왔다.
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, replace
from datetime import datetime

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.recommend_reason import context_reason
from app.components.reranker import TrackCandidate
from app.components.scene_curator import Scene, SceneCurator
from app.components.scene_picker import MIN_TASTE_CANDIDATES, Pick, pick_scene_songs
from app.components.song_curator import SongCandidate
from app.components.song_mix import resolve_sources
from app.components.song_pool import SongPool
from app.components.taste_genres import OnboardingReader, no_onboarding, taste_genres
from app.components.taste_vector import blend_taste, mean_vector
from app.components.track_resolver import TrackResolver
from app.components.vector_search import RecordSearch
from app.config import SIMILARITY_DEFAULTS
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.exceptions import ExternalApiFailedError, LLMError, UpstreamError
from app.schemas.context_recommend import (
    ContextRecommendRequest,
    ContextRecommendResponse,
    RankedTrack,
)

logger = logging.getLogger("muro.context_recommend")

# "비슷한 과거 코멘트"로 인정하는 최소 유사도의 기본값. 임베딩 모델마다 점수 분포가 달라서
# 실제 값은 설정(personal_similarity_min, 없으면 모델별 기본값)에서 받는다 (app/config.py 참고).
PERSONAL_SIMILARITY_THRESHOLD = SIMILARITY_DEFAULTS["kure"]
SIMILAR_SEARCH_LIMIT = 5
REFERENCE_COUNT = 3
# 주변 인기곡 중 벡터를 찾아볼 최대 곡 수
NEARBY_VECTOR_LIMIT = 20
# 후보·최종곡을 로그로 남길 때 보여 줄 후보 수
LOG_CANDIDATES = 10
# Spotify에서 못 찾는 곡이 있어도 5곡이 채워지도록 더 뽑아 두는 여분 (백엔드는 정확히 5곡만 받는다)
SPARE_PICKS = 5
# 2026-10-02 팀 결정: 추천은 최대 5곡. 백엔드가 limit을 더 크게 보내도 5곡까지만 준다.
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

SEASONS = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}  # fmt: skip


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
        scene_curator: SceneCurator,
        spotify_search: SpotifySearchClient,
        track_lookups: TrackLookupStore | None = None,
        song_pool: SongPool | None = None,
        personal_similarity: float = PERSONAL_SIMILARITY_THRESHOLD,
        nearby_weight: float = 0.0,
        hub_weight: float = 0.0,
        onboarding: OnboardingReader = no_onboarding,
        rng: random.Random | None = None,
    ):
        self._embedder = embedder
        self._records = record_search
        self._scene_curator = scene_curator
        self._spotify = spotify_search
        self._resolver = TrackResolver(spotify_search, track_lookups)
        self._pool = song_pool
        self._personal_similarity = personal_similarity
        self._nearby_weight = nearby_weight
        self._hub_weight = hub_weight
        self._onboarding = onboarding
        self._rng = rng or random.Random()

    async def recommend(self, req: ContextRecommendRequest) -> ContextRecommendResponse:
        moment = f"{_weather_phrase(req.weather.condition)} {_time_bucket(req.local_time)}"
        references = await self._personal_references(req.user_id, moment)
        basis = _basis(references, req)
        situation = _situation_text(req)

        if self._pool is None:
            logger.warning("곡풀이 없어 추천할 수 없음, 폴백")
            return _degraded(req, basis)
        try:
            scenes = await self._scene_curator.for_context(situation)
            lyric_queries, description_queries = await self._scene_vectors(scenes)
        except LLMError as e:
            logger.warning("장면 문장 만들기 실패, 폴백: %s", e)
            return _degraded(req, basis)
        except Exception:
            logger.warning("장면 문장 임베딩 실패, 폴백", exc_info=True)
            return _degraded(req, basis)
        lyric_queries = [self._with_nearby(req, v) for v in lyric_queries]

        # 과거에 들은 곡(참고로 쓴 곡)과 재추천 때 클라이언트가 보낸 곡을 함께 뺀다
        excluded = {r.external_track_id for r in references} | set(req.exclude_track_ids)
        heard = [SongCandidate(_first_artist(r.artist), r.title, "", ()) for r in references]
        ranked = self._pool.rank_scenes(
            lyric_queries, description_queries, exclude=heard, hub_weight=self._hub_weight
        )
        if not ranked:
            logger.warning("곡풀 점수를 매기지 못함 (벡터 길이 불일치?), 폴백")
            return _degraded(req, basis)

        taste = await taste_genres(req.user_id, self._onboarding, self._records)
        limit = min(req.limit, MAX_TRACKS)
        # 곡풀 곡은 Spotify ID를 확인한 뒤에야 알아서, 재추천 때 뺄 곡이 섞인 만큼과 Spotify에서
        # 못 찾을 곡을 위한 여분(SPARE_PICKS)을 더 뽑아 둔다. 확인한 곡은 track_lookups에 남아
        # 다음부터 Spotify 호출이 늘지 않는다
        picks = pick_scene_songs(
            ranked,
            limit + len(excluded) + SPARE_PICKS,
            taste,
            outside_position=limit - 1,
            rng=self._rng,
        )
        _log_picks(situation, taste, ranked, picks, scenes)

        songs = [replace(p.song.to_candidate(), score=p.score) for p in picks]
        try:
            _, found = await resolve_sources(self._resolver, [], songs, [])
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패: %s", e)
            raise ExternalApiFailedError("Spotify 곡 확인에 실패했어요.") from e
        kept = [c for c in found if c.external_track_id not in excluded][:limit]

        return ContextRecommendResponse(
            request_id=req.request_id,
            rewritten_query=situation,
            recommendation_basis=basis,
            tracks=[_to_ranked_track(c) for c in kept],
            recommend_reason=context_reason(
                _when_phrase(req),
                [p.song.genres[0] for p in picks[:limit] if p.song.genres],
                taste if _taste_used(ranked, taste) else None,
            ),
        )

    async def _scene_vectors(
        self, scenes: list[Scene]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """장면마다 가사 문체 문장·설명 문체 문장을 벡터로 (한 번의 배치 호출)."""
        vectors = await self._embedder.embed_batch(
            [s.lyric for s in scenes] + [s.description for s in scenes]
        )
        return vectors[: len(scenes)], vectors[len(scenes) :]

    def _with_nearby(self, req: ContextRecommendRequest, vector: list[float]) -> list[float]:
        """주변 사람들이 많이 담은 곡 중 곡풀에 있는 곡의 벡터(담긴 횟수로 가중 평균)를 섞는다."""
        if self._nearby_weight <= 0 or self._pool is None or not req.nearby_tracks:
            return vector
        nearby = req.nearby_tracks[:NEARBY_VECTOR_LIMIT]
        found = self._pool.vectors_for([(t.artist_name, t.title) for t in nearby])
        if not found:
            return vector
        weights = [float(nearby[i].count or 1) for i, _ in found]
        mean = mean_vector([list(embedding) for _, embedding in found], weights)
        return blend_taste(vector, mean, self._nearby_weight)

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
            if hit.similarity < self._personal_similarity or not hit.external_track_id:
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


def _situation_text(req: ContextRecommendRequest) -> str:
    """ "겨울(12월), 평일 저녁, 비 오는 날 (기온 10도)" — 장면 문장 실험에서 쓴 것과 같은 모양."""
    t = req.local_time
    daytype = "주말" if t.weekday() >= 5 else "평일"
    text = (
        f"{SEASONS[t.month]}({t.month}월), {daytype} {_time_bucket(t)}, "
        f"{_weather_phrase(req.weather.condition)} 날"
    )
    if req.weather.temperature is not None:
        text += f" (기온 {req.weather.temperature:g}도)"
    if req.place and req.place.name:
        text += f", 장소: {req.place.name}"
    return text


def _when_phrase(req: ContextRecommendRequest) -> str:
    """추천 이유에 쓰는 상황 말 — "비 오는 겨울 평일 저녁". 모르는 날씨 값은 뺀다."""
    t = req.local_time
    daytype = "주말" if t.weekday() >= 5 else "평일"
    weather = WEATHER_PHRASES.get(req.weather.condition.strip().upper())
    parts = [weather, SEASONS[t.month], daytype, _time_bucket(t)]
    return " ".join(p for p in parts if p)


def _taste_used(ranked: list, taste: list[str]) -> bool:
    """취향 장르로 실제로 걸렀는지 — scene_picker와 같은 기준(후보가 충분할 때만 거른다)."""
    if not taste:
        return False
    wanted = set(taste)
    return sum(1 for r in ranked if wanted & set(r.song.genres)) >= MIN_TASTE_CANDIDATES


def _log_picks(
    situation: str, taste: list[str], ranked: list, picks: list[Pick], scenes: list[Scene]
) -> None:
    """dev에서 후보와 최종 곡이 어떻게 갈렸는지 볼 수 있게 남긴다 (가사는 남기지 않는다)."""
    top = ", ".join(
        f"{r.song.artist}-{r.song.title}({r.score:.2f})" for r in ranked[:LOG_CANDIDATES]
    )
    final = ", ".join(
        f"{p.song.artist}-{p.song.title}({p.score:.2f},장면{p.scene_index + 1}"
        f"{',취향밖' if p.outside_taste else ''})"
        for p in picks
    )
    logger.info(
        "추천 상황=%s 취향장르=%s 장면=%s 상위후보=[%s] 최종=[%s]",
        situation,
        taste or "없음",
        [s.scene for s in scenes],
        top,
        final,
    )


def _to_ranked_track(c: TrackCandidate) -> RankedTrack:
    return RankedTrack(
        title=c.title,
        artist_name=c.artist_name,
        external_track_id=c.external_track_id,
        spotify_uri=c.spotify_uri,
        album_image_url=c.album_image_url,
        external_url=c.external_url,
        rerank_score=round(c.score if c.score is not None else 0.0, 3),
    )


def _degraded(req: ContextRecommendRequest, basis: str) -> ContextRecommendResponse:
    return ContextRecommendResponse(
        request_id=req.request_id,
        rewritten_query="",
        recommendation_basis=basis,
        tracks=[],
        degraded=True,
    )
