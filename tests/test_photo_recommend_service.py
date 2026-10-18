import pytest

from app.clients.spotify_client import SpotifyError
from app.components.clip_tagger import PhotoAnalysis
from app.components.reranker import Reranker
from app.components.song_curator import CurationResult, SongCandidate
from app.exceptions import ClipError, ExternalApiFailedError, LLMError
from app.schemas.photo_recommend import PhotoRecommendRequest
from app.services.photo_recommend_service import PhotoRecommendService


class StubClipTagger:
    def __init__(
        self,
        tags: list[str] | None = None,
        error: Exception | None = None,
        axes: dict[str, str] | None = None,
    ):
        self._tags = tags if tags is not None else []
        self._error = error
        self._axes = axes or {}

    async def analyze(self, image_url: str, top_k: int = 3) -> PhotoAnalysis:
        if self._error:
            raise self._error
        return PhotoAnalysis(scene_tags=self._tags, axes=self._axes)


class StubSongCurator:
    """첫 호출은 curation, 곡이 모자라 보충할 때(exclude 있음)는 extra를 돌려준다."""

    def __init__(
        self,
        curation: CurationResult | None = None,
        error: Exception | None = None,
        extra: CurationResult | None = None,
    ):
        self._curation = curation if curation is not None else _curation()
        self._error = error
        self._extra = extra or CurationResult((), (), [])
        self.excludes: list[list[SongCandidate]] = []
        self.axes_seen: list[dict[str, str] | None] = []
        self.counts: list[int] = []

    async def curate_from_tags(
        self,
        tags: list[str],
        exclude: list[SongCandidate] | None = None,
        axes: dict[str, str] | None = None,
        count: int = 10,
    ) -> CurationResult:
        self.axes_seen.append(axes)
        self.counts.append(count)
        if exclude:
            self.excludes.append(exclude)
            return self._extra
        if self._error:
            raise self._error
        return self._curation


class StubSpotifySearch:
    def __init__(
        self,
        results_by_query: dict[str, list[dict] | Exception],
        error: Exception | None = None,
        tracks_by_id: dict[str, dict | Exception] | None = None,
    ):
        self._results = results_by_query
        self._error = error
        self._tracks = tracks_by_id or {}
        self.queries: list[str] = []
        self.track_ids: list[str] = []

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        self.queries.append(query)
        if self._error:
            raise self._error
        result = self._results.get(query, [])
        if isinstance(result, Exception):
            raise result
        return result

    async def get_track(self, track_id: str) -> dict:
        self.track_ids.append(track_id)
        result = self._tracks.get(track_id)
        if result is None:
            raise KeyError(track_id)
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
        situation_genres=(genre,),
        situation_moods=moods,
        songs=list(songs) if songs is not None else [_song("아이유", "밤편지")],
    )


def _req(**overrides) -> PhotoRecommendRequest:
    payload = {"imageUrl": "https://cdn.muro.app/uploads/img_1.jpg"}
    payload.update(overrides)
    return PhotoRecommendRequest.model_validate(payload)


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


async def test_only_top_5_are_returned_even_if_more_resolve():
    songs = [_song("A", f"Song {i}") for i in range(7)]
    results = {_q("A", f"Song {i}"): [_track_json(f"t{i}", f"Song {i}", "A")] for i in range(7)}
    service = build_service(curation=_curation(songs=songs), results=results)

    res = await service.recommend(_req())

    assert len(res.tracks) == 5


async def test_tops_up_once_when_fewer_than_5_resolve():
    """Spotify에서 탈락해 5곡이 안 되면 이미 나온 곡을 빼고 한 번 더 받아 채운다."""
    first = [_song("A", f"Song {i}") for i in range(2)]
    extra = [_song("B", f"More {i}") for i in range(3)]
    results = {_q("A", f"Song {i}"): [_track_json(f"t{i}", f"Song {i}", "A")] for i in range(2)}
    results |= {_q("B", f"More {i}"): [_track_json(f"m{i}", f"More {i}", "B")] for i in range(3)}
    curator = StubSongCurator(curation=_curation(songs=first), extra=_curation(songs=extra))
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"]),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(results),
    )

    res = await service.recommend(_req())

    assert len(res.tracks) == 5
    assert curator.excludes == [first]


async def test_top_up_failure_keeps_what_we_have():
    first = [_song("A", "Song 0")]
    results = {_q("A", "Song 0"): [_track_json("t0", "Song 0", "A")]}

    class TopUpFails(StubSongCurator):
        async def curate_from_tags(self, tags, exclude=None, axes=None, count=10):
            if exclude:
                raise LLMError("장애")
            return _curation(songs=first)

    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"]),
        song_curator=TopUpFails(),
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(results),
    )

    res = await service.recommend(_req())

    assert [t.external_track_id for t in res.tracks] == ["t0"]


async def test_searches_spotify_with_title_and_artist_field_filters():
    spotify = StubSpotifySearch({_q("아이유", "밤편지"): [_track_json("t1", "밤편지", "아이유")]})
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"]),
        song_curator=StubSongCurator(curation=_curation(songs=[_song("아이유", "밤편지")])),
        reranker=Reranker(),
        spotify_search=spotify,
    )

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


async def test_all_songs_spotify_failure_raises_502():
    """전부 오류로 실패하면(=Spotify 자체 장애로 추정) 빈 200 대신 502로 알린다."""
    songs = [_song("A", "Song A"), _song("B", "Song B")]
    results = {
        _q("A", "Song A"): SpotifyError("장애"),
        _q("B", "Song B"): SpotifyError("장애"),
    }
    service = build_service(curation=_curation(songs=songs), results=results)

    with pytest.raises(ExternalApiFailedError):
        await service.recommend(_req())


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


AXES = {"사진 종류": "진짜 사진", "대상": "여러 사람", "분위기": "밤의 따뜻한 조명"}


async def test_axes_go_to_the_song_curator_but_mood_tags_stay_scene_only():
    """다축 태그는 Claude에 보내는 정보에만 쓴다 — 백엔드로 가는 moodTags는 장소 태그 그대로."""
    curator = StubSongCurator(curation=_curation(songs=[_song("아이유", "밤편지")]))
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"], axes=AXES),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(
            {_q("아이유", "밤편지"): [_track_json("t1", "밤편지", "아이유")]}
        ),
    )

    res = await service.recommend(_req())

    assert res.mood_tags == ["카페"]
    assert curator.axes_seen[0] == AXES


async def test_top_up_call_also_carries_the_axes():
    first = [_song("A", "Song 0")]
    extra = [_song("B", f"More {i}") for i in range(4)]
    results = {_q("A", "Song 0"): [_track_json("t0", "Song 0", "A")]}
    results |= {_q("B", f"More {i}"): [_track_json(f"m{i}", f"More {i}", "B")] for i in range(4)}
    curator = StubSongCurator(curation=_curation(songs=first), extra=_curation(songs=extra))
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"], axes=AXES),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(results),
    )

    await service.recommend(_req())

    assert curator.axes_seen == [AXES, AXES]  # 첫 추천과 보충 추천 둘 다


async def test_without_axes_the_curator_gets_none_like_before():
    curator = StubSongCurator()
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["카페"]),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotifySearch({}),
    )

    await service.recommend(_req())

    assert curator.axes_seen[0] == {}


# ── 곡 풀 섞기 (5곡 = 곡 풀 3 + LLM 2) ─────────────────────────────


def _pool_of(*songs):
    import random

    from app.components.song_pool import PoolSong, SongPool

    return SongPool(
        [PoolSong(a, t, ("댄스",), ("신나는", "화려한")) for a, t in songs],
        rng=random.Random(0),
    )


def _results_for(*pairs):
    return {_q(a, t): [_track_json(f"id-{a}-{t}", t, a)] for a, t in pairs}


def build_mixed(*, llm_songs, pool_songs, results, pool_count=3, llm_count=4):
    curator = StubSongCurator(curation=_curation(songs=[_song(a, t) for a, t in llm_songs]))
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["파티"]),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotifySearch(results),
        song_pool=_pool_of(*pool_songs),
        pool_count=pool_count,
        llm_count=llm_count,
    )
    return service, curator


async def test_mixes_three_pool_songs_and_two_llm_songs():
    llm = [("L1", "a"), ("L2", "b"), ("L3", "c"), ("L4", "d")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z"), ("P4", "w")]
    service, curator = build_mixed(
        llm_songs=llm, pool_songs=pool, results=_results_for(*llm, *pool)
    )

    res = await service.recommend(_req())

    artists = [t.artist_name for t in res.tracks]
    assert len(artists) == 5
    assert sum(a.startswith("P") for a in artists) == 3
    assert sum(a.startswith("L") for a in artists) == 2
    # 곡 풀을 쓰면 LLM에게는 적게 받는다
    assert curator.counts == [4]


async def test_llm_fills_when_pool_has_no_matching_songs():
    llm = [("L1", "a"), ("L2", "b"), ("L3", "c"), ("L4", "d"), ("L5", "e")]
    service, _ = build_mixed(llm_songs=llm, pool_songs=[], results=_results_for(*llm))

    res = await service.recommend(_req())

    assert [t.artist_name for t in res.tracks] and all(
        t.artist_name.startswith("L") for t in res.tracks
    )
    assert len(res.tracks) == 5


async def test_pool_fills_when_llm_songs_are_not_on_spotify():
    llm = [("L1", "a"), ("L2", "b"), ("L3", "c"), ("L4", "d")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z"), ("P4", "w"), ("P5", "v")]
    # LLM 곡은 하나만 Spotify에 있다
    service, _ = build_mixed(
        llm_songs=llm, pool_songs=pool, results=_results_for(("L1", "a"), *pool), pool_count=5
    )

    res = await service.recommend(_req())

    assert len(res.tracks) == 5


async def test_pool_disabled_by_zero_count_keeps_old_behavior():
    llm = [("L1", "a")]
    service, curator = build_mixed(
        llm_songs=llm,
        pool_songs=[("P1", "x")],
        results=_results_for(*llm, ("P1", "x")),
        pool_count=0,
    )

    res = await service.recommend(_req())

    assert [t.artist_name for t in res.tracks] == ["L1"]
    assert curator.counts[0] == 10


async def test_pool_spotify_failure_still_answers_with_llm_songs():
    llm = [("L1", "a"), ("L2", "b")]
    results = {**_results_for(*llm), _q("P1", "x"): SpotifyError("pool down")}
    service, _ = build_mixed(llm_songs=llm, pool_songs=[("P1", "x")], results=results)

    res = await service.recommend(_req())

    assert {t.artist_name for t in res.tracks} == {"L1", "L2"}


async def test_both_spotify_failures_raise_502():
    results = {_q("L1", "a"): SpotifyError("down"), _q("P1", "x"): SpotifyError("down")}
    service, _ = build_mixed(llm_songs=[("L1", "a")], pool_songs=[("P1", "x")], results=results)

    with pytest.raises(ExternalApiFailedError):
        await service.recommend(_req())


# ── 사용자 저장 곡을 곡 풀과 한 풀로 (2026-10-16) ───────────────────────


def _saved_pool_of(*track_ids, genre="댄스", moods=("신나는", "화려한")):
    import random

    from app.components.saved_song_pool import SavedSongPool
    from app.db.repositories.track_mood_repository import SavedSong

    songs = [SavedSong(tid, genre, tuple(moods)) for tid in track_ids]

    class _Search:
        async def search_by_tags(self, genres, moods_, limit):
            return songs

    return SavedSongPool(_Search(), rng=random.Random(0))


def build_with_saved(*, llm_songs, pool_songs, saved_ids, saved_count=1, pool_count=3):
    results = _results_for(*llm_songs, *pool_songs)
    tracks = {tid: _track_json(tid, f"제목-{tid}", f"가수-{tid}") for tid in saved_ids}
    curator = StubSongCurator(curation=_curation(songs=[_song(a, t) for a, t in llm_songs]))
    spotify = StubSpotifySearch(results, tracks_by_id=tracks)
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["파티"]),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=spotify,
        song_pool=_pool_of(*pool_songs),
        pool_count=pool_count,
        llm_count=4,
        saved_pool=_saved_pool_of(*saved_ids),
        saved_count=saved_count,
    )
    return service, spotify


async def test_saved_song_takes_one_of_the_pool_slots():
    llm = [("L1", "a"), ("L2", "b"), ("L3", "c"), ("L4", "d")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z")]
    service, spotify = build_with_saved(llm_songs=llm, pool_songs=pool, saved_ids=["s1"])

    res = await service.recommend(_req())

    ids = [t.external_track_id for t in res.tracks]
    assert len(ids) == 5
    assert "s1" in ids  # 저장 곡 1곡
    assert sum(i.startswith("id-P") for i in ids) == 2  # 곡 풀이 남은 2자리
    # 저장 곡은 검색이 아니라 get_track 한 번으로 끝난다
    assert spotify.track_ids == ["s1"]


async def test_song_pool_fills_the_slot_when_no_saved_song_matches():
    llm = [("L1", "a"), ("L2", "b")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z")]
    service, spotify = build_with_saved(llm_songs=llm, pool_songs=pool, saved_ids=[])

    res = await service.recommend(_req())

    ids = [t.external_track_id for t in res.tracks]
    assert sum(i.startswith("id-P") for i in ids) == 3  # 저장 곡 자리까지 곡 풀이 메움
    assert spotify.track_ids == []


async def test_saved_song_lookup_failure_still_answers_with_the_rest():
    llm = [("L1", "a"), ("L2", "b")]
    pool = [("P1", "x"), ("P2", "y")]
    results = _results_for(*llm, *pool)
    curator = StubSongCurator(curation=_curation(songs=[_song(a, t) for a, t in llm]))
    spotify = StubSpotifySearch(results, tracks_by_id={"s1": SpotifyError("down")})
    service = PhotoRecommendService(
        clip_tagger=StubClipTagger(tags=["파티"]),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=spotify,
        song_pool=_pool_of(*pool),
        pool_count=3,
        llm_count=4,
        saved_pool=_saved_pool_of("s1"),
        saved_count=1,
    )

    res = await service.recommend(_req())

    ids = [t.external_track_id for t in res.tracks]
    assert "s1" not in ids
    assert ids  # 나머지로 응답
