"""기능3 처리 순서: 사진 내려받기 → Gemma가 사진을 직접 보고 장면 3개·문장 2종 → 곡풀 벡터 점수
(허브 보정) → 상위 곡 중 무작위 → Spotify 실존 확인 → 5곡.

2026-11: CLIP 태그 단계를 뺐다. 예전에는 CLIP이 장소·표정 태그를 뽑고 상용 LLM(Claude)이 태그를 곡
추천용 분위기 문장으로 바꿨다. 지금은 RunPod의 Gemma(사진 입력 가능)가 사진을 직접 보고 장면을
상상해, 장면마다 가사 문체·설명 문체 문장을 쓴다. 곡풀의 가사 벡터·곡 설명 벡터와 같은 문체끼리
비교해 점수를 내고(장면 3개 평균), 상위 곡에서 무작위로 5곡을 뽑는다(scene_picker.py).
실측(아카이브 사진 47장, 허브 보정 0.5, Gemma가 사진·곡을 보고 1~5점): CLIP 태그 방식 3.20,
사진 직접 보기 3.50, 사진+CLIP 태그 같이 주기 3.39 — CLIP 태그를 더해도 나아지지 않았다.
CLIP은 기능4(자물쇠 저장)에서 사진 벡터와 장소 태그를 만들 때만 쓴다 (RECAP이 그걸 쓴다).
기능3은 사용자 ID를 받지 않아서 취향 장르로 거르지 않는다.

에러 처리 원칙(위키 5-1절): 사진·LLM 장애는 폴백 + degraded:true, 결과 0건은 에러가 아니라
빈 배열(degraded:false). Spotify 장애는 502 — 곡을 하나도 확인할 수 없어 200 빈 결과로는 백엔드가
장애를 알아챌 수 없었다. moodTags는 CLIP 태그를 뺐으므로 항상 빈 목록이다 (백엔드는 이 칸을
읽지 않는다 — recommendReason·tracks·degraded만 쓴다).

(참고) 재추천 제외(excludeTrackIds)는 기능1(상황 기반)에만 있다 — 기능3은 사진 한
장당 한 번 추천하는 구조라 재추천 동작 자체가 없다.

※ "지역 인기곡 폴백"은 백엔드에 해당 조회 API(FR-008 집계)가 아직 없어
  place만 받아두고 실제 조회는 미구현 — 지금은 빈 tracks로 대체한다.
"""

import logging
import random
from dataclasses import replace

from app.clients.image_fetcher import ImageFetcher, ImageFetchError
from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.recommend_reason import photo_reason
from app.components.reranker import TrackCandidate
from app.components.scene_curator import Scene, SceneCurator
from app.components.scene_picker import Pick, pick_scene_songs
from app.components.song_mix import resolve_sources
from app.components.song_pool import SongPool
from app.components.track_resolver import TrackResolver
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.exceptions import ExternalApiFailedError, LLMError, UpstreamError
from app.schemas.common import Track
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse

logger = logging.getLogger("muro.photo_recommend")

RESULT_COUNT = 5
LOG_CANDIDATES = 10


class PhotoRecommendService:
    def __init__(
        self,
        scene_curator: SceneCurator,
        spotify_search: SpotifySearchClient,
        embedder: Embedder,
        image_fetcher: ImageFetcher,
        track_lookups: TrackLookupStore | None = None,
        song_pool: SongPool | None = None,
        hub_weight: float = 0.0,
        rng: random.Random | None = None,
    ):
        self._scene_curator = scene_curator
        self._resolver = TrackResolver(spotify_search, track_lookups)
        self._embedder = embedder
        self._images = image_fetcher
        self._pool = song_pool
        self._hub_weight = hub_weight
        self._rng = rng or random.Random()

    async def recommend(self, req: PhotoRecommendRequest) -> PhotoRecommendResponse:
        if self._pool is None:
            logger.warning("곡풀이 없어 추천할 수 없음, 폴백")
            return self._fallback()
        try:
            image = await self._images.fetch_data_url(req.image_url)
        except ImageFetchError as e:
            logger.warning("사진을 받지 못해 폴백: %s", e)
            return self._fallback()

        try:
            reading = await self._scene_curator.for_photo(image)
            scenes = reading.scenes
            vectors = await self._embedder.embed_batch(
                [s.lyric for s in scenes] + [s.description for s in scenes]
            )
        except LLMError as e:
            logger.warning("장면 문장 만들기 실패, 폴백: %s", e)
            return self._fallback()
        except Exception:
            logger.warning("장면 문장 임베딩 실패, 폴백", exc_info=True)
            return self._fallback()

        ranked = self._pool.rank_scenes(
            vectors[: len(scenes)], vectors[len(scenes) :], hub_weight=self._hub_weight
        )
        if not ranked:
            logger.warning("곡풀 점수를 매기지 못함 (벡터 길이 불일치?), 폴백")
            return self._fallback()
        picks = pick_scene_songs(ranked, RESULT_COUNT, rng=self._rng)
        _log_picks(ranked, picks, scenes, reading.summary)

        songs = [replace(p.song.to_candidate(), score=p.score) for p in picks]
        try:
            _, found = await resolve_sources(self._resolver, [], songs, [])
        except UpstreamError as e:
            logger.warning("Spotify 확인 실패: %s", e)
            raise ExternalApiFailedError("Spotify 곡 확인에 실패했어요.") from e

        if not found:
            # 뽑은 곡이 전부 Spotify에 없었다 — 에러는 아니고 빈 결과
            return PhotoRecommendResponse(mood_tags=[], rewritten_query="", tracks=[])
        return PhotoRecommendResponse(
            mood_tags=[],
            rewritten_query="",
            tracks=[_to_track(c) for c in found[:RESULT_COUNT]],
            recommend_reason=photo_reason(
                [p.song.genres[0] for p in picks if p.song.genres], reading.summary
            ),
        )

    def _fallback(self) -> PhotoRecommendResponse:
        return PhotoRecommendResponse(mood_tags=[], rewritten_query="", tracks=[], degraded=True)


def _log_picks(ranked: list, picks: list[Pick], scenes: list[Scene], summary: str) -> None:
    """dev에서 후보와 최종 곡이 어떻게 갈렸는지 볼 수 있게 남긴다 (가사는 남기지 않는다)."""
    top = ", ".join(
        f"{r.song.artist}-{r.song.title}({r.score:.2f})" for r in ranked[:LOG_CANDIDATES]
    )
    final = ", ".join(
        f"{p.song.artist}-{p.song.title}({p.score:.2f},장면{p.scene_index + 1})" for p in picks
    )
    logger.info(
        "사진 추천 요약=%s 장면=%s 상위후보=[%s] 최종=[%s]",
        summary or "없음",
        [s.scene for s in scenes],
        top,
        final,
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
