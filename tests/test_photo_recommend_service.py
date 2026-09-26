from app.clients.spotify_client import SpotifyError
from app.components.reranker import Reranker
from app.components.song_curator import CurationResult, SongCandidate
from app.exceptions import ClipError, LLMError
from app.schemas.photo_recommend import PhotoRecommendRequest
from app.services.photo_recommend_service import PhotoRecommendService


class StubClipTagger:
    def __init__(self, tags: list[str] | None = None, error: Exception | None = None):
        self._tags = tags if tags is not None else []
        self._error = error

    async def tag(self, image_url: str, top_k: int = 3) -> list[str]:
        if self._error:
            raise self._error
        return self._tags


class StubSongCurator:
    def __init__(self, curation: CurationResult | None = None, error: Exception | None = None):
        self._curation = curation if curation is not None else _curation()
        self._error = error

    async def curate_from_tags(self, tags: list[str]) -> CurationResult:
        if self._error:
            raise self._error
        return self._curation


class StubSpotifySearch:
    def __init__(
        self,
        results_by_query: dict[str, list[dict] | Exception],
        error: Exception | None = None,
    ):
        self._results = results_by_query
        self._error = error
        self.queries: list[str] = []

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        self.queries.append(query)
        if self._error:
            raise self._error
        result = self._results.get(query, [])
        if isinstance(result, Exception):
            raise result
        return result


def _track_json(track_id: str, title: str, artist: str) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": artist}],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": [{"url": f"https://img/{track_id}.jpg"}]},
    }


def _song(artist: str, title: str, genre: str = "댄스", moods=("신나는",)) -> SongCandidate:
    return SongCandidate(artist=artist, title=title, genre=genre, moods=moods)


def _q(artist: str, title: str) -> str:
    return _song(artist, title).search_query


def _curation(songs=None, genre="댄스", moods=("신나는", "화려한")) -> CurationResult:
    return CurationResult(
        situation_genre=genre,
        situation_moods=moods,
        songs=list(songs) if songs is not None else [_song("아이유", "밤편지")],
    )


def _req() -> PhotoRecommendRequest:
    return PhotoRecommendRequest(imageUrl="https://cdn.muro.app/uploads/img_1.jpg")


def build_service(*, tags=("카페",), curation=None, results=None):
    return PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=list(tags)),
        song_curator=StubSongCurator(curation=curation),
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(results if results is not None else {}),
    )


async def test_happy_path_returns_top_3_tracks_ranked_by_tag_overlap():
    songs = [
        _song("아이유", "밤편지", genre="발라드", moods=("잔잔한", "그리운")),
        _song("The Weeknd", "Blinding Lights", genre="댄스", moods=("화려한", "신나는")),
        _song("지코", "아무노래", genre="힙합", moods=("신나는", "경쾌한")),
    ]
    curation = _curation(songs=songs, genre="댄스", moods=("신나는", "화려한"))
    results = {
        _q("아이유", "밤편지"): [_track_json("t1", "밤편지", "아이유")],
        _q("The Weeknd", "Blinding Lights"): [_track_json("t2", "Blinding Lights", "The Weeknd")],
        _q("지코", "아무노래"): [_track_json("t3", "아무노래", "지코")],
    }
    service = build_service(curation=curation, results=results)

    res = await service.recommend(_req())

    assert res.mood_tags == ["카페"]
    assert res.degraded is False
    # 댄스/화려한+신나는과 제일 잘 맞는 The Weeknd가 1등으로 올라와야 한다
    assert res.tracks[0].title == "Blinding Lights"


async def test_only_top_3_are_returned_even_if_more_resolve():
    songs = [_song("A", f"Song {i}") for i in range(5)]
    results = {_q("A", f"Song {i}"): [_track_json(f"t{i}", f"Song {i}", "A")] for i in range(5)}
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert len(res.tracks) == 3


async def test_searches_spotify_with_title_and_artist_field_filters():
    service = build_service(curation=_curation(songs=[_song("아이유", "밤편지")]))
    spotify = service._spotify

    await service.recommend(_req())

    assert spotify.queries == ["track:밤편지 artist:아이유"]


async def test_songs_missing_on_spotify_are_dropped():
    """LLM이 없는 곡을 지어내도(할루시네이션) 확인 단계에서 걸러진다."""
    songs = [_song("아이유", "밤편지"), _song("없는가수", "없는곡")]
    results = {_q("아이유", "밤편지"): [_track_json("t1", "밤편지", "아이유")]}
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert [t.title for t in res.tracks] == ["밤편지"]
    assert res.degraded is False


async def test_one_song_spotify_failure_does_not_fail_the_whole_request():
    """8곡 중 1곡만 Spotify 오류가 나도 나머지로 계속 진행한다."""
    songs = [_song("아이유", "밤편지"), _song("장애곡", "장애곡")]
    results = {
        _q("아이유", "밤편지"): [_track_json("t1", "밤편지", "아이유")],
        _q("장애곡", "장애곡"): SpotifyError("일시 장애"),
    }
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert res.degraded is False
    assert [t.title for t in res.tracks] == ["밤편지"]


async def test_all_songs_spotify_failure_falls_back_degraded():
    """전부 오류로 실패하면(=Spotify 자체 장애로 추정) 폴백으로 처리한다."""
    songs = [_song("A", "Song A"), _song("B", "Song B")]
    results = {
        _q("A", "Song A"): SpotifyError("장애"),
        _q("B", "Song B"): SpotifyError("장애"),
    }
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert res.degraded is True


async def test_duplicate_tracks_are_deduplicated():
    songs = [_song("아이유", "밤편지"), _song("IU", "Through the Night")]
    same_track = _track_json("t1", "밤편지", "아이유")
    results = {_q("아이유", "밤편지"): [same_track], _q("IU", "Through the Night"): [same_track]}
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert len(res.tracks) == 1


async def test_clip_failure_falls_back_degraded():
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(error=ClipError("장애")),
        song_curator=StubSongCurator(),
        reranker=Reranker(),
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


async def test_curation_failure_falls_back_with_tags_kept():
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"]),
        song_curator=StubSongCurator(error=LLMError("장애")),
        reranker=Reranker(),
        spotify_search=StubSpotifySearch({}),
    )

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.mood_tags == ["카페"]


async def test_unparseable_curation_falls_back_degraded():
    service = build_service(curation=_curation(songs=[]))

    res = await service.recommend(_req())

    assert res.degraded is True
    assert res.mood_tags == ["카페"]


async def test_all_songs_missing_is_empty_but_not_degraded():
    """추천곡이 전부 Spotify에 없어도 장애는 아니다 — 빈 배열로 응답한다."""
    service = build_service(curation=_curation(songs=[_song("없는가수", "없는곡")]), results={})

    res = await service.recommend(_req())

    assert res.degraded is False
    assert res.tracks == []
