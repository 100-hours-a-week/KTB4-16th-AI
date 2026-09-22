from app.clients.spotify_client import SpotifyError
from app.exceptions import ClipError, LLMError
from app.services.photo_recommend_service import PhotoRecommendService


class StubClipTagger:
    def __init__(self, tags: list[str] | None = None, error: Exception | None = None):
        self._tags = tags if tags is not None else []
        self._error = error

    async def tag(self, image_url: str, top_k: int = 3) -> list[str]:
        if self._error:
            raise self._error
        return self._tags


class StubQueryRewriter:
    def __init__(self, query: str = "", error: Exception | None = None):
        self._query = query
        self._error = error

    async def rewrite_from_tags(self, tags: list[str]) -> str:
        if self._error:
            raise self._error
        return self._query


class StubSpotifySearch:
    def __init__(self, results_by_query: dict[str, list[dict]], error: Exception | None = None):
        self._results = results_by_query
        self._error = error
        self.queries: list[str] = []

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        self.queries.append(query)
        if self._error:
            raise self._error
        return self._results.get(query, [])


class PassthroughReranker:
    async def rerank(self, query_text, candidates):
        return candidates


class BrokenReranker:
    async def rerank(self, query_text, candidates):
        raise RuntimeError("재랭킹 인프라 장애")


def _track_json(track_id: str, title: str, artist: str) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": artist}],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": [{"url": f"https://img/{track_id}.jpg"}]},
    }


def build_service(*, tags=("노을",), query="노을 시티팝", results=None, reranker=None):
    results = results if results is not None else {}
    return PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=list(tags)),
        query_rewriter=StubQueryRewriter(query=query),
        reranker=reranker or PassthroughReranker(),
        spotify_search=StubSpotifySearch(results),
    )


async def test_happy_path_returns_top_3_tracks():
    tracks_json = [_track_json(f"t{i}", f"Song {i}", "Artist") for i in range(5)]
    service = build_service(results={"노을 시티팝": tracks_json})

    res = await service.recommend(_req())

    assert res.mood_tags == ["노을"]
    assert res.rewritten_query == "노을 시티팝"
    assert res.degraded is False
    assert len(res.tracks) == 3
    assert res.tracks[0].title == "Song 0"
    assert res.retry_count == 0  # 첫 검색어로 바로 성공


async def test_clip_failure_falls_back_degraded():
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(error=ClipError("장애")),
        query_rewriter=StubQueryRewriter(),
        reranker=PassthroughReranker(),
        spotify_search=StubSpotifySearch({}),
    )

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.mood_tags == []
    assert res.tracks == []


async def test_no_tags_above_threshold_falls_back_degraded():
    service = build_service(tags=())

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.tracks == []


async def test_rewrite_failure_falls_back_with_tags_kept():
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["노을"]),
        query_rewriter=StubQueryRewriter(error=LLMError("장애")),
        reranker=PassthroughReranker(),
        spotify_search=StubSpotifySearch({}),
    )

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.mood_tags == ["노을"]  # CLIP까지는 성공했으니 태그는 유지
    assert res.rewritten_query == ""


async def test_search_failure_falls_back_with_query_kept():
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["노을"]),
        query_rewriter=StubQueryRewriter(query="노을 시티팝"),
        reranker=PassthroughReranker(),
        spotify_search=StubSpotifySearch({}, error=SpotifyError("장애")),
    )

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.rewritten_query == "노을 시티팝"


async def test_zero_results_after_relax_retry_is_not_degraded():
    """검색 0건은 스펙상 에러가 아니다 — 완화 재시도도 실패하면 그냥 빈 배열."""
    service = build_service(query="노을 시티팝", results={})

    res = await service.recommend(_req())

    assert res.degraded is False
    assert res.tracks == []
    assert res.retry_count == 1  # 완화 재시도는 시도했음 (결과가 0건이었을 뿐)


async def test_zero_results_retries_with_last_word_of_query():
    tracks_json = [_track_json("t1", "시티팝 명곡", "Artist")]
    spotify = StubSpotifySearch({"시티팝": tracks_json})  # "노을 시티팝"으로는 0건
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["노을"]),
        query_rewriter=StubQueryRewriter(query="노을 시티팝"),
        reranker=PassthroughReranker(),
        spotify_search=spotify,
    )

    res = await service.recommend(_req())

    assert spotify.queries == ["노을 시티팝", "시티팝"]
    assert len(res.tracks) == 1
    assert res.degraded is False
    assert res.retry_count == 1


async def test_reranker_failure_keeps_original_order_without_failing_request():
    tracks_json = [_track_json("t1", "Song 1", "Artist")]
    service = build_service(results={"노을 시티팝": tracks_json}, reranker=BrokenReranker())

    res = await service.recommend(_req())

    assert res.degraded is False
    assert len(res.tracks) == 1


def _req():
    from app.schemas.photo_recommend import PhotoRecommendRequest

    return PhotoRecommendRequest(imageUrl="https://cdn.muro.app/uploads/img_1.jpg")


async def test_single_word_query_has_nothing_to_relax_so_no_retry():
    """검색어가 이미 한 단어면 완화할 게 없다 — 재시도 자체를 안 하고 retryCount=0."""
    service = build_service(query="시티팝", results={})

    res = await service.recommend(_req())

    assert res.retry_count == 0
    assert res.tracks == []
