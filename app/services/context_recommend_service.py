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

2026-10-16: 곡 풀 몫 중 일부(기본 1곡)는 사용자들이 자물쇠에 저장한 곡(track_moods)에서
가져온다 — TJ 고정 목록과 저장 곡으로 곡 풀이 흩어지지 않게 같은 점수 기준으로 묶어 한
풀처럼 쓴다(saved_song_pool.py). 저장 곡은 이미 Spotify ID가 있어 검색이 아니라 get_track
한 번으로 끝난다.

재추천: 클라이언트가 excludeTrackIds로 직전까지 보여준 곡을 누적해서 보내면 그 곡들을
빼고 추천한다(개인 취향 참고곡 제외와 같은 방식으로 합쳐서 처리). 서버는 상태를 들고
있지 않는다.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime

from app.clients.spotify_client import SpotifySearchClient
from app.components.db_picker import DbPickConfig, pick_db_songs
from app.components.embedder import Embedder
from app.components.reranker import Reranker, TrackCandidate
from app.components.saved_song_pool import SavedSongPool
from app.components.song_curator import (
    CANDIDATE_COUNT,
    CurationResult,
    SongCandidate,
    SongCurator,
)
from app.components.song_mix import resolve_sources
from app.components.song_pool import SongPool
from app.components.taste_vector import blend_taste, mean_vector
from app.components.track_resolver import TrackResolver
from app.components.vector_search import RecordSearch
from app.config import SIMILARITY_DEFAULTS
from app.db.repositories.memory_record_repository import MemoryRecordSource
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.db.repositories.track_mood_repository import SavedSong
from app.exceptions import ExternalApiFailedError, LLMError, UpstreamError
from app.schemas.context_recommend import (
    ContextRecommendRequest,
    ContextRecommendResponse,
    RankedTrack,
)

logger = logging.getLogger("muro.context_recommend")

# "비슷한 과거 코멘트"로 인정하는 최소 유사도의 기본값. 임베딩 모델마다 점수 분포가 달라서
# 실제 값은 설정(personal_similarity_min, 없으면 모델별 기본값)에서 받는다 (app/config.py 참고).
# OpenAI 시절 실측(2026-09-26 더미 자물쇠): 비슷한 코멘트 0.40~0.74, 상관없는 코멘트 0.26~0.39.
# KURE 값은 합성 시험셋으로 잡은 임시값 — 실제 사용자 코멘트가 쌓이면 다시 맞춰야 한다.
PERSONAL_SIMILARITY_THRESHOLD = SIMILARITY_DEFAULTS["kure"]
SIMILAR_SEARCH_LIMIT = 5
# 취향 벡터를 만들 때 쓰는 최근 저장 곡 수
TASTE_RECORD_LIMIT = 50
# 상황(날씨)이 맞는 기록을 찾으려고 최근 기록을 읽는 수
SITUATIONAL_ROW_LIMIT = 300
# 주변 인기곡 중 벡터를 찾아볼 최대 곡 수
NEARBY_VECTOR_LIMIT = 20
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
        saved_pool: SavedSongPool | None = None,
        saved_count: int = 0,
        personal_similarity: float = PERSONAL_SIMILARITY_THRESHOLD,
        taste_weight: float = 0.0,
        record_source: MemoryRecordSource | None = None,
        situational_min_records: int = 3,
        nearby_weight: float = 0.0,
        db_pick: DbPickConfig = DbPickConfig(),
    ):
        self._db_pick = db_pick
        self._taste_weight = taste_weight
        self._record_source = record_source
        self._situational_min_records = situational_min_records
        self._nearby_weight = nearby_weight
        self._saved_pool = saved_pool
        self._saved_count = saved_count if saved_pool is not None else 0
        self._personal_similarity = personal_similarity
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

        # 과거에 들은 곡(취향 참고로 쓴 곡)과 재추천 때 클라이언트가 보낸 곡을 함께 뺀다
        excluded = {r.external_track_id for r in references} | set(req.exclude_track_ids)
        situation_vector = await self._mood_vector(curation.mood_sentence)
        situation_vector = await self._bias_vector(req, situation_vector)
        saved_songs, pool_songs = await self._pick_db(
            curation, references, excluded, situation_vector
        )
        if not curation.songs and not pool_songs and not saved_songs:
            logger.warning("곡 추천 결과를 파싱하지 못함, 폴백")
            return _degraded(req, basis)

        try:
            llm_found, pool_found = await resolve_sources(
                self._resolver, curation.songs, pool_songs, saved_songs
            )
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패: %s", e)
            raise ExternalApiFailedError("Spotify 곡 확인에 실패했어요.") from e

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
        """태그 점수 순으로 줄 세우고 기준 미만은 뺀다 (Claude 곡처럼 벡터가 없는 곡용)."""
        ranked = self._reranker.rank(
            curation.situation_genres, curation.situation_moods, candidates
        )
        return [(c, score) for c, score in ranked if score >= MIN_RERANK_SCORE]

    def _keep_db(
        self, curation: CurationResult, candidates: list[TrackCandidate]
    ) -> list[tuple[TrackCandidate, float]]:
        """DB 곡: 곡을 고를 때 받은 하이브리드 점수(태그+벡터) 순. 점수가 없는 곡은 태그 점수로."""
        scored = [(c, c.score) for c in candidates if c.score is not None]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        unscored = self._keep(curation, [c for c in candidates if c.score is None])
        return scored + unscored

    def _mix(
        self,
        curation: CurationResult,
        llm_found: list[TrackCandidate],
        pool_found: list[TrackCandidate],
        limit: int,
    ) -> list[tuple[TrackCandidate, float]]:
        """DB 곡(저장된 곡+곡 풀) pool_count곡을 하이브리드 점수 순으로, 그 뒤에 Claude 곡을 둔다.

        Claude 곡은 벡터가 없어 DB 곡과 같은 점수로 비교할 수 없으니 순서를 겨루지 않고 뒤에 둔다.
        한쪽이 모자라면 다른 쪽으로 채운다.
        """
        pool_kept = self._keep_db(curation, pool_found)
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
        return mixed

    async def _pick_db(
        self,
        curation: CurationResult,
        references: list["ReferenceTrack"],
        excluded: set[str],
        situation_vector: list[float] | None,
    ) -> tuple[list[SavedSong], list[SongCandidate]]:
        """저장된 곡과 곡 풀을 같은 점수(태그+벡터)로 겨뤄 DB 몫만큼 고른다 (db_picker)."""
        total = self._pool_count if self._pool is not None else self._saved_count
        if total <= 0:
            return [], []
        heard = [SongCandidate(_first_artist(r.artist), r.title, "", ()) for r in references]
        pool_ranked = []
        if self._pool is not None:
            try:
                pool_ranked = self._pool.ranked(
                    curation.situation_genres,
                    curation.situation_moods,
                    exclude=curation.songs + heard,
                    situation_vector=situation_vector,
                )
            except Exception:
                logger.exception("곡 풀에서 고르기 실패 — 저장된 곡·LLM 곡만 사용")
        saved_ranked = []
        if self._saved_pool is not None:
            saved_ranked = await self._saved_pool.ranked(
                curation.situation_genres,
                curation.situation_moods,
                exclude_ids=excluded,
                situation_vector=situation_vector,
            )
        config = replace(
            self._db_pick,
            min_saved=min(self._saved_count, total),
            max_saved=max(self._db_pick.max_saved, self._saved_count),
        )
        return pick_db_songs(saved_ranked, pool_ranked, total, config)

    async def _mood_vector(self, sentence: str) -> list[float] | None:
        """Claude가 쓴 분위기 문장을 벡터로. 실패해도 태그 점수로 고르면 되니 추천은 계속한다."""
        if not sentence or (self._pool is None and self._saved_pool is None):
            return None
        try:
            return await self._embedder.embed(sentence)
        except Exception:
            logger.warning("분위기 문장 임베딩 실패 — 태그 점수로만 고름", exc_info=True)
            return None

    async def _bias_vector(
        self, req: ContextRecommendRequest, situation_vector: list[float] | None
    ) -> list[float] | None:
        """상황 벡터를 내 취향·주변 곡 쪽으로 조금 옮긴다 (곡을 고를 때 쓰는 기준점).

        Claude 프롬프트가 아니라 코드에서 비율을 정한다. 둘 다 실패하거나 재료가 없으면
        상황 벡터 그대로다.
        """
        if situation_vector is None:
            return None
        vector = situation_vector
        if self._taste_weight > 0:
            vector = blend_taste(vector, await self._taste_vector(req), self._taste_weight)
        return self._with_nearby(req, vector)

    async def _taste_vector(self, req: ContextRecommendRequest) -> list[float] | None:
        """지금과 같은 날씨에 저장한 기록이 충분하면 그 기록들로, 아니면 최근 저장 곡 전체로."""
        try:
            situational = await self._situational_taste(req)
            if situational is not None:
                return situational
            return mean_vector(await self._records.music_vectors(req.user_id, TASTE_RECORD_LIMIT))
        except Exception:
            logger.warning("취향 벡터 조회 실패 — 상황 벡터만 사용", exc_info=True)
            return None

    async def _situational_taste(self, req: ContextRecommendRequest) -> list[float] | None:
        if self._record_source is None:
            return None
        weather = req.weather.condition.strip().upper()
        rows = await self._record_source.list_condition_rows(
            user_id=req.user_id, limit=SITUATIONAL_ROW_LIMIT
        )
        record_ids = [
            r.record_id
            for r in rows
            if r.weather_condition and r.weather_condition.strip().upper() == weather
        ]
        if len(record_ids) < self._situational_min_records:
            return None
        vectors = await self._records.music_vectors_for_records(req.user_id, record_ids)
        if len(vectors) < self._situational_min_records:
            return None
        return mean_vector(vectors)

    def _with_nearby(self, req: ContextRecommendRequest, vector: list[float]) -> list[float]:
        """주변 사람들이 많이 담은 곡 중 곡 풀에 있는 곡의 벡터(담긴 횟수로 가중 평균)를 섞는다."""
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
