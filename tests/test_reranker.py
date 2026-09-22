from app.components.embedder import Embedder
from app.components.reranker import BACKFILL_JOB_TYPE, Reranker, TrackCandidate
from app.db.repositories.track_mood_repository import CachedMood


class ScriptedEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]):
        self.model = "scripted-embed"
        self.dim = 3
        self._vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vectors[t] for t in texts]


class FakeTrackMoodStore:
    def __init__(self, cached: dict[str, CachedMood] | None = None):
        self._cached = cached or {}

    async def get_any(self, external_track_id: str) -> CachedMood | None:
        return self._cached.get(external_track_id)


class FakeJobs:
    def __init__(self, fail_for: set[str] | None = None):
        self.enqueued: list[dict] = []
        self._fail_for = fail_for or set()

    async def enqueue(self, *, job_type, dedupe_key, payload, max_attempts) -> int:
        if dedupe_key in self._fail_for:
            raise RuntimeError("boom")
        self.enqueued.append({"job_type": job_type, "dedupe_key": dedupe_key, "payload": payload})
        return len(self.enqueued)


def _candidate(track_id: str, title: str, artist: str, popularity: int = 0) -> TrackCandidate:
    return TrackCandidate(
        external_track_id=track_id,
        title=title,
        artist_name=artist,
        spotify_uri=f"spotify:track:{track_id}",
        popularity=popularity,
    )


QUERY = "노을, 밤거리"
CAND_A = _candidate("a1", "Song A", "Artist A")  # 캐시 있음, 질의와 완전히 같은 방향
CAND_B = _candidate("b1", "Song B", "Artist B")  # 캐시 없음, 질의와 직교
CAND_C = _candidate("c1", "Song C", "Artist C")  # 캐시 없음, 질의와 거의 같은 방향

VECTORS = {
    QUERY: [1.0, 0.0, 0.0],
    "Song B Artist B": [0.0, 1.0, 0.0],
    "Song C Artist C": [0.9, 0.1, 0.0],
}


def _reranker(client, track_moods, jobs) -> Reranker:
    return Reranker(embedder=Embedder(client), track_moods=track_moods, jobs=jobs)


async def test_empty_candidates_short_circuits():
    reranker = _reranker(ScriptedEmbeddingClient({}), FakeTrackMoodStore(), FakeJobs())
    assert await reranker.rerank(QUERY, []) == []


async def test_rerank_orders_by_similarity_mixing_cache_and_miss():
    client = ScriptedEmbeddingClient(VECTORS)
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})
    jobs = FakeJobs()

    ranked = await _reranker(client, track_moods, jobs).rerank(QUERY, [CAND_B, CAND_A, CAND_C])

    assert [c.external_track_id for c in ranked] == ["a1", "c1", "b1"]


async def test_cache_misses_are_embedded_in_a_single_batch_call():
    client = ScriptedEmbeddingClient(VECTORS)
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})

    await _reranker(client, track_moods, FakeJobs()).rerank(QUERY, [CAND_A, CAND_B, CAND_C])

    # 1번은 query_text 임베딩(단건), 1번은 캐시 미스 2건 배치 — 곡마다 호출하지 않는다
    assert client.calls == [[QUERY], ["Song B Artist B", "Song C Artist C"]]


async def test_cache_misses_enqueue_backfill_but_hits_do_not():
    jobs = FakeJobs()
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})

    await _reranker(ScriptedEmbeddingClient(VECTORS), track_moods, jobs).rerank(
        QUERY, [CAND_A, CAND_B, CAND_C]
    )

    queued_ids = {job["dedupe_key"] for job in jobs.enqueued}
    assert queued_ids == {"b1", "c1"}
    assert all(job["job_type"] == BACKFILL_JOB_TYPE for job in jobs.enqueued)


async def test_backfill_enqueue_failure_does_not_break_reranking():
    jobs = FakeJobs(fail_for={"b1"})
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})

    ranked = await _reranker(ScriptedEmbeddingClient(VECTORS), track_moods, jobs).rerank(
        QUERY, [CAND_A, CAND_B, CAND_C]
    )

    assert [c.external_track_id for c in ranked] == ["a1", "c1", "b1"]
    assert {job["dedupe_key"] for job in jobs.enqueued} == {"c1"}  # b1은 실패, c1만 등록됨


async def test_high_popularity_can_overtake_slightly_lower_mood_match():
    """무드만 보면 A가 근소 우위지만, 인기도가 압도적이면 B가 역전한다.

    mood: A=1.0(캐시) vs B=0.9(임베딩) / popularity: A=0 vs B=100
    score = 0.7*mood + 0.3*(popularity/100)
      A = 0.7*1.0 + 0.3*0.0 = 0.70
      B = 0.7*0.9 + 0.3*1.0 = 0.93
    """
    unpopular = _candidate("a1", "Song A", "Artist A", popularity=0)
    popular = _candidate("b1", "Song B", "Artist B", popularity=100)
    client = ScriptedEmbeddingClient({**VECTORS, "Song B Artist B": [0.9, 0.1, 0.0]})
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})

    ranked = await _reranker(client, track_moods, FakeJobs()).rerank(QUERY, [unpopular, popular])

    assert [c.external_track_id for c in ranked] == ["b1", "a1"]


async def test_equal_popularity_falls_back_to_pure_mood_order():
    """인기도가 같으면(기본값 0) 지금까지의 무드 순서 그대로 유지 — 회귀 방지."""
    client = ScriptedEmbeddingClient(VECTORS)
    track_moods = FakeTrackMoodStore({"a1": CachedMood("설명", [1.0, 0.0, 0.0])})

    ranked = await _reranker(client, track_moods, FakeJobs()).rerank(
        QUERY, [CAND_B, CAND_A, CAND_C]
    )

    assert [c.external_track_id for c in ranked] == ["a1", "c1", "b1"]
