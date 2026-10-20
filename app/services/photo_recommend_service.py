"""기능3 처리 순서: CLIP 태깅 → LLM 곡+태그 추천 → Spotify 실존 확인 → 재랭킹 → 상위 5곡.

이전엔 LLM이 만든 장르 키워드로 Spotify를 넓게 검색하고 그 결과를 재랭킹으로 걸렀다.
그다음엔 LLM이 실제 곡을 직접 추천하고 임베딩 유사도로 재랭킹했는데, 자유 문장
임베딩이 "신나는지 잔잔한지" 같은 축을 신뢰성 있게 구분 못 했다(자세한 배경은
song_curator.py, reranker.py). 그래서 지금은 song_curator가 곡 추천과 동시에
사진 상황·각 곡의 장르·무드 태그까지 같은 호출에서 받고, 재랭킹은 그 태그가
얼마나 겹치는지로 점수를 매긴다.

에러 처리 원칙(위키 5-1절): CLIP·LLM 장애는 폴백 + degraded:true, 결과 0건은 에러가
아니라 빈 배열(degraded:false). Spotify 장애는 502 — 곡을 하나도 확인할 수 없어
200 빈 결과로는 백엔드가 장애를 알아챌 수 없었다. 재랭킹 실패는 후보 자체는
유효하므로 순서만 원본으로 두고 응답은 그대로 낸다.

2026-10-15: 5곡 중 일부(기본 3곡)를 곡 풀(song_pool.py — 노래방 인기 차트 기반, 장르·무드
태그를 미리 붙인 곡 목록)에서 채운다. LLM이 고르는 곡은 그만큼 적게(기본 4곡) 받는다.
곡 풀은 LLM이 준 "상황 장르·무드"로 고르므로 LLM 호출은 늘지 않고, LLM 출력과 Spotify
검색은 오히려 줄어든다. 곡 풀이 없거나 맞는 곡이 모자라면 LLM 곡으로 채운다.

2026-10-20: Claude가 같은 호출에서 쓴 "분위기 문장"(사진에서 뽑은 모든 정보를 녹인 문장)을 벡터로
바꿔서, 곡 풀·저장된 곡을 장르·무드로 거른 뒤 그 벡터와 가까운 순으로 정렬한다 (기능1과 같은
방식). 사진 정보가 장르 2~3개·무드 2~3개로 줄어들면서 사라지는 것을 벡터가 보태 준다. 기능3은
사용자 ID를 받지 않아서 취향 섞기는 없다. 임베딩이 실패해도 예전 태그 점수 순으로 계속한다.

2026-10-16: 곡 풀 몫 중 일부(기본 1곡)는 사용자들이 자물쇠에 저장한 곡(track_moods)에서
가져온다 — 곡 풀이 TJ 고정 목록과 저장 곡 두 개로 흩어지지 않게 같은 점수 기준으로 묶어
한 풀처럼 쓴다(saved_song_pool.py).

(참고) 재추천 제외(excludeTrackIds)는 기능1(상황 기반)에만 있다 — 기능3은 사진 한
장당 한 번 추천하는 구조라 재추천 동작 자체가 없다.

※ "지역 인기곡 폴백"은 백엔드에 해당 조회 API(FR-008 집계)가 아직 없어
  place만 받아두고 실제 조회는 미구현 — 지금은 빈 tracks로 대체한다.
"""

import logging
from dataclasses import replace

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.db_picker import DbPickConfig, pick_db_songs
from app.components.embedder import Embedder
from app.components.reranker import Reranker, TrackCandidate
from app.components.saved_song_pool import SavedSongPool
from app.components.song_curator import CANDIDATE_COUNT, CurationResult, SongCandidate, SongCurator
from app.components.song_mix import resolve_sources
from app.components.song_pool import SongPool
from app.components.track_resolver import TrackResolver
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.db.repositories.track_mood_repository import SavedSong
from app.exceptions import ClipError, ExternalApiFailedError, LLMError, UpstreamError
from app.schemas.common import Track
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse

logger = logging.getLogger("muro.photo_recommend")

RESULT_COUNT = 5


class PhotoRecommendService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        song_curator: SongCurator,
        reranker: Reranker,
        spotify_search: SpotifySearchClient,
        track_lookups: TrackLookupStore | None = None,
        song_pool: SongPool | None = None,
        pool_count: int = 0,
        llm_count: int = CANDIDATE_COUNT,
        saved_pool: SavedSongPool | None = None,
        saved_count: int = 0,
        embedder: Embedder | None = None,
        db_pick: DbPickConfig = DbPickConfig(),
    ):
        self._db_pick = db_pick
        self._embedder = embedder
        self._saved_pool = saved_pool
        self._saved_count = saved_count if saved_pool is not None else 0
        self._clip_tagger = clip_tagger
        self._song_curator = song_curator
        self._reranker = reranker
        self._resolver = TrackResolver(spotify_search, track_lookups)
        self._pool = song_pool if song_pool is not None and pool_count > 0 else None
        self._pool_count = pool_count if self._pool is not None else 0
        # 곡 풀을 안 쓰면 예전처럼 LLM에게 넉넉히 받는다
        self._llm_count = llm_count if self._pool is not None else CANDIDATE_COUNT

    async def recommend(self, req: PhotoRecommendRequest) -> PhotoRecommendResponse:
        try:
            analysis = await self._clip_tagger.analyze(req.image_url)
        except ClipError as e:
            logger.warning("CLIP 태깅 실패, 폴백: %s", e)
            return self._fallback()

        # moodTags(백엔드 응답)는 장소 태그만 담는다. 다축 태그는 Claude에 보내는 정보에만 쓴다.
        tags, axes = analysis.scene_tags, analysis.axes
        if not tags:
            # 임계치 넘는 태그가 하나도 없음 — 곡을 고를 근거가 없어 폴백과 동일 처리
            return self._fallback()

        try:
            curation = await self._song_curator.curate_from_tags(
                tags, axes=axes, count=self._llm_count
            )
        except LLMError as e:
            logger.warning("곡 추천 실패, 폴백: %s", e)
            return self._fallback(tags=tags)

        situation_vector = await self._mood_vector(curation.mood_sentence)
        saved_songs, pool_songs = await self._pick_db(curation, situation_vector)
        if not curation.songs and not pool_songs and not saved_songs:
            # 형식이 깨져 한 줄도 못 건졌다 — 추천 자체가 실패한 것으로 본다
            logger.warning("곡 추천 결과를 파싱하지 못함, 폴백")
            return self._fallback(tags=tags)

        try:
            llm_found, pool_found = await resolve_sources(
                self._resolver, curation.songs, pool_songs, saved_songs
            )
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패: %s", e)
            raise ExternalApiFailedError("Spotify 곡 확인에 실패했어요.") from e
        candidates = self._mix(curation, llm_found, pool_found)

        if len(candidates) < RESULT_COUNT:
            candidates = await self._top_up(tags, axes, curation, candidates)

        if not candidates:
            # 추천받은 곡이 전부 Spotify에 없었다 — 에러는 아니고 빈 결과
            return PhotoRecommendResponse(mood_tags=tags, rewritten_query="", tracks=[])

        # 구성(DB 곡·Claude 곡 몫)과 순서(DB 곡은 점수 순, Claude 곡은 뒤)는 _mix에서 정했다
        return PhotoRecommendResponse(
            mood_tags=tags,
            rewritten_query="",
            tracks=[_to_track(c) for c in candidates[:RESULT_COUNT]],
        )

    async def _top_up(
        self,
        tags: list[str],
        axes: dict[str, str],
        curation: CurationResult,
        candidates: list[TrackCandidate],
    ) -> list[TrackCandidate]:
        """Spotify에서 탈락해 RESULT_COUNT곡이 안 되면 이미 나온 곡을 빼고 한 번만 더 받는다.

        보충은 있으면 좋은 것이라, 여기서 실패해도 지금 있는 곡으로 응답한다.
        """
        try:
            extra = await self._song_curator.curate_from_tags(
                tags, exclude=curation.songs, axes=axes, count=self._llm_count
            )
            more = await self._resolver.resolve(extra.songs)
        except UpstreamError as e:
            logger.warning("곡 보충 실패, 있는 곡으로 응답: %s", e)
            return candidates
        seen = {c.external_track_id for c in candidates}
        return candidates + [c for c in more if c.external_track_id not in seen]

    async def _mood_vector(self, sentence: str) -> list[float] | None:
        """Claude가 쓴 분위기 문장을 벡터로. 실패해도 태그 점수로 고르면 되니 추천은 계속한다."""
        if (
            self._embedder is None
            or not sentence
            or (self._pool is None and self._saved_pool is None)
        ):
            return None
        try:
            return await self._embedder.embed(sentence)
        except Exception:
            logger.warning("분위기 문장 임베딩 실패 — 태그 점수로만 고름", exc_info=True)
            return None

    async def _pick_db(
        self, curation: CurationResult, situation_vector: list[float] | None
    ) -> tuple[list[SavedSong], list[SongCandidate]]:
        """저장된 곡과 곡 풀을 같은 점수(태그+벡터)로 겨뤄 DB 몫만큼 고른다 (db_picker)."""
        total = self._pool_count if self._pool is not None else self._saved_count
        if total <= 0:
            return [], []
        pool_ranked = []
        if self._pool is not None:
            try:
                pool_ranked = self._pool.ranked(
                    curation.situation_genres,
                    curation.situation_moods,
                    exclude=curation.songs,
                    situation_vector=situation_vector,
                )
            except Exception:
                logger.exception("곡 풀에서 고르기 실패 — 저장된 곡·LLM 곡만 사용")
        saved_ranked = []
        if self._saved_pool is not None:
            saved_ranked = await self._saved_pool.ranked(
                curation.situation_genres,
                curation.situation_moods,
                situation_vector=situation_vector,
            )
        config = replace(
            self._db_pick,
            min_saved=min(self._saved_count, total),
            max_saved=max(self._db_pick.max_saved, self._saved_count),
        )
        return pick_db_songs(saved_ranked, pool_ranked, total, config)

    def _mix(
        self,
        curation: CurationResult,
        llm_found: list[TrackCandidate],
        pool_found: list[TrackCandidate],
    ) -> list[TrackCandidate]:
        """DB 곡 pool_count곡을 하이브리드 점수 순으로, 그 뒤에 Claude 곡(태그 점수 순)을 둔다.

        Claude 곡은 벡터가 없어 DB 곡과 같은 점수로 비교할 수 없으니 순서를 겨루지 않고 뒤에 둔다.
        한쪽이 모자라면 다른 쪽으로 채운다.
        """
        llm_ranked = self._rerank_safely(curation, llm_found)
        pool_ranked = self._rank_db(curation, pool_found)
        pool_first = pool_ranked[: self._pool_count]
        ordered = (
            pool_first
            + llm_ranked[: RESULT_COUNT - len(pool_first)]
            + pool_ranked[self._pool_count :]
            + llm_ranked[RESULT_COUNT - len(pool_first) :]
        )
        mixed: list[TrackCandidate] = []
        seen: set[str] = set()
        for c in ordered:
            if c.external_track_id in seen:
                continue
            seen.add(c.external_track_id)
            mixed.append(c)
        return mixed

    def _rank_db(
        self, curation: CurationResult, candidates: list[TrackCandidate]
    ) -> list[TrackCandidate]:
        """DB 곡: 곡을 고를 때 받은 하이브리드 점수 순. 점수가 없는 곡은 태그 점수로 뒤에 붙인다."""
        scored = sorted(
            (c for c in candidates if c.score is not None),
            key=lambda c: c.score,
            reverse=True,
        )
        return scored + self._rerank_safely(curation, [c for c in candidates if c.score is None])

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
