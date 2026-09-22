"""후보곡 재랭킹 — 기능3(사진 무드) 사용, 기능1(재랭킹 점수 가중합)이 이어서 재사용.

Spotify 검색은 문자열 매칭이라 검색 결과가 실제로 분위기와 안 맞을 수 있다
(실측: "노을 지는 바다가 떠오르는 시티팝"으로 검색했더니 시티팝이 하나도
없었음). 그래서 검색은 장르로 넓게 하고, 후보곡을 무드로 다시 줄세운다.

**공간 정합성이 핵심이다.** 사진의 CLIP 벡터는 CLIP 공간, 후보곡의
트랙무드 벡터(기능4에서 만든 것)는 텍스트 임베딩(OpenAI) 공간이라 서로
비교할 수 없다. 그래서 재랭킹은 CLIP을 쓰지 않고, "질의 텍스트"와
"후보곡 텍스트"를 **같은 임베딩 모델로** 인코딩해서 비교한다.

**지연 예산 때문에 LLM은 이 동기 경로에서 쓰지 않는다.** 후보 10~20곡마다
LLM으로 트랙무드를 새로 만들면 수십 초가 걸린다(기능3은 "수 초 이내" 요구).
그래서:
  1) 트랙무드 캐시(기능4가 다른 자물쇠로 이미 만들어둔 것)가 있으면 재사용
  2) 없으면 "제목+아티스트" 원문을 임베딩 API로 배치 호출 1번에 벡터화
     (LLM 없이, 정확도는 낮지만 빠르고 쌈)
  3) 그 곡들은 백그라운드 작업으로 "진짜" 트랙무드 생성을 걸어둔다
     (기능4 워커가 처리 — 다음에 같은 곡이 후보로 나오면 1)로 해결됨)

**점수는 무드 유사도만 보지 않는다.** 실측 결과 무드 유사도만으로 정렬하면
무명 곡이 우연히 벡터가 잘 맞아 1위로 올라오는 경우가 있었다. 위키 1단계
설계(기능1 재랭킹: "유사도·인기도·무드 매칭 가중합")에 이미 있던 인기도를
반영해, Spotify가 주는 popularity(0~100)를 섞어 최종 점수를 낸다.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.components.embedder import Embedder
from app.components.vector_math import cosine_similarity
from app.db.repositories.track_mood_repository import TrackMoodStore

logger = logging.getLogger("muro.reranker")

BACKFILL_JOB_TYPE = "track_mood_backfill"
BACKFILL_MAX_ATTEMPTS = 3


# 무드 유사도 : 인기도 가중치. 인기도를 너무 높이면 사진 분위기와 상관없이
# 그냥 유명한 곡만 나오므로, 무드를 주된 신호로 두고 인기도는 보조로만 쓴다.
MOOD_WEIGHT = 0.7
POPULARITY_WEIGHT = 0.3


@dataclass(frozen=True)
class TrackCandidate:
    external_track_id: str
    title: str
    artist_name: str
    spotify_uri: str
    album_art_url: str | None = None
    popularity: int = 0  # Spotify 제공값 0~100, 못 받으면 0(가장 불리하게)


class JobEnqueuer(Protocol):
    async def enqueue(
        self, *, job_type: str, dedupe_key: str, payload: dict[str, Any], max_attempts: int
    ) -> int: ...


class Reranker:
    def __init__(self, embedder: Embedder, track_moods: TrackMoodStore, jobs: JobEnqueuer):
        self._embedder = embedder
        self._track_moods = track_moods
        self._jobs = jobs

    async def rerank(
        self, query_text: str, candidates: list[TrackCandidate]
    ) -> list[TrackCandidate]:
        if not candidates:
            return []

        query_vector = await self._embedder.embed(query_text)
        vectors = await self._resolve_vectors(candidates)

        return sorted(candidates, key=lambda c: self._score(c, query_vector, vectors), reverse=True)

    def _score(
        self, c: TrackCandidate, query_vector: list[float], vectors: dict[str, list[float]]
    ) -> float:
        mood_score = cosine_similarity(query_vector, vectors[c.external_track_id])
        popularity_score = c.popularity / 100
        return MOOD_WEIGHT * mood_score + POPULARITY_WEIGHT * popularity_score

    async def _resolve_vectors(self, candidates: list[TrackCandidate]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        misses: list[TrackCandidate] = []
        for c in candidates:
            cached = await self._track_moods.get_any(c.external_track_id)
            if cached is not None:
                vectors[c.external_track_id] = cached.mood_embedding
            else:
                misses.append(c)

        if misses:
            raw_vectors = await self._embedder.embed_batch(
                [f"{c.title} {c.artist_name}" for c in misses]
            )
            for c, v in zip(misses, raw_vectors, strict=True):
                vectors[c.external_track_id] = v
            await self._enqueue_backfill(misses)

        return vectors

    async def _enqueue_backfill(self, misses: list[TrackCandidate]) -> None:
        """캐시 없던 곡들의 '진짜' 트랙무드를 백그라운드로 만들어두게 큐에 등록.

        실패해도 지금 응답에는 영향 없어야 하므로 예외를 삼킨다.
        """
        for c in misses:
            try:
                await self._jobs.enqueue(
                    job_type=BACKFILL_JOB_TYPE,
                    dedupe_key=c.external_track_id,
                    payload={
                        "externalTrackId": c.external_track_id,
                        "title": c.title,
                        "artistName": c.artist_name,
                    },
                    max_attempts=BACKFILL_MAX_ATTEMPTS,
                )
            except Exception:
                logger.exception("트랙무드 백필 등록 실패: %s", c.external_track_id)
