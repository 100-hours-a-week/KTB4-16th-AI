from datetime import datetime

import pytest

from app.clients.spotify_client import SpotifyError
from app.components.reranker import Reranker
from app.components.song_curator import CurationResult, SongCandidate
from app.components.vector_search import SimilarRecord
from app.exceptions import ExternalApiFailedError, LLMError
from app.schemas.context_recommend import ContextRecommendRequest
from app.services.context_recommend_service import (
    NEARBY_REFERENCE_COUNT,
    PERSONAL_SIMILARITY_THRESHOLD,
    ContextRecommendService,
    _time_bucket,
    _weather_phrase,
)


class StubEmbedder:
    model = "stub-embed"

    def __init__(self, error: Exception | None = None):
        self._error = error
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        if self._error:
            raise self._error
        return [1.0, 0.0, 0.0]


class StubRecordSearch:
    def __init__(self, hits: list[SimilarRecord] | None = None, error: Exception | None = None):
        self._hits = hits or []
        self._error = error
        self.calls: list[int] = []

    async def similar_by_comment(self, user_id, query_vector, limit):
        self.calls.append(user_id)
        if self._error:
            raise self._error
        return self._hits


class StubSongCurator:
    """첫 호출은 curation, 곡이 모자라 보충할 때(exclude 있음)는 extra를 돌려준다."""

    def __init__(
        self,
        curation: CurationResult | None = None,
        error: Exception | None = None,
        extra: CurationResult | None = None,
    ):
        self._curation = curation
        self._error = error
        self._extra = extra or CurationResult((), (), [])
        self.situations: list[str] = []
        self.excludes: list[list[SongCandidate]] = []
        self.counts: list[int] = []

    async def curate(
        self, situation: str, exclude: list[SongCandidate] | None = None, count: int = 10
    ) -> CurationResult:
        self.counts.append(count)
        if exclude:
            self.excludes.append(exclude)
            return self._extra
        self.situations.append(situation)
        if self._error:
            raise self._error
        return self._curation


class StubSpotify:
    def __init__(self, search: dict | None = None, tracks: dict | None = None):
        self._search = search or {}
        self._tracks = tracks or {}

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        result = self._search.get(query, [])
        if isinstance(result, Exception):
            raise result
        return result

    async def get_track(self, track_id: str) -> dict:
        return self._tracks[track_id]


def _track_json(track_id: str, title: str, artist: str) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": artist}],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": [{"url": f"https://i.scdn.co/image/{track_id}"}]},
    }


def _song(artist, title, genre="발라드", moods=("잔잔한",)) -> SongCandidate:
    return SongCandidate(artist=artist, title=title, genre=genre, moods=moods)


def _curation(songs, genres=("발라드",), moods=("잔잔한", "그리운")) -> CurationResult:
    return CurationResult(situation_genres=genres, situation_moods=moods, songs=list(songs))


def _req(**overrides) -> ContextRecommendRequest:
    payload = {
        "requestId": "req_1",
        "userId": 1,
        "place": {"placeId": 10, "name": "동탄역 카페거리"},
        "weather": {"condition": "RAIN", "temperature": 18},
        "localTime": "2026-08-31T19:40:00+09:00",
        "limit": 3,
    }
    payload.update(overrides)
    return ContextRecommendRequest.model_validate(payload)


def _service(*, curator, spotify=None, records=None, embedder=None, **extra):
    return ContextRecommendService(
        embedder=embedder or StubEmbedder(),
        record_search=records or StubRecordSearch(),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=spotify or StubSpotify(),
        **extra,
    )


BALLAD = _song("아이유", "밤편지")
DEFAULT_SEARCH = {BALLAD.search_query: [_track_json("t_ballad", "밤편지", "아이유")]}


async def test_situation_prompt_has_weather_time_bucket_temperature_and_place():
    curator = StubSongCurator(_curation([BALLAD]))

    await _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH)).recommend(_req())

    assert curator.situations[0] == "상황: 비 오는 저녁, 기온 18도, 장소: 동탄역 카페거리"


async def test_personal_search_uses_only_weather_and_time_bucket():
    """장소 이름은 과거 코멘트에 거의 안 나와 유사도만 흐려서 검색 질의에서 뺀다."""
    embedder = StubEmbedder()
    curator = StubSongCurator(_curation([BALLAD]))

    await _service(
        curator=curator, embedder=embedder, spotify=StubSpotify(DEFAULT_SEARCH)
    ).recommend(_req())

    assert embedder.texts == ["비 오는 저녁"]


async def test_generic_when_no_similar_past_record():
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH)).recommend(_req())

    assert res.recommendation_basis == "GENERIC"
    assert "들었던 곡" not in curator.situations[0]
    assert [t.title for t in res.tracks] == ["밤편지"]


async def test_personal_when_similar_past_record_passes_threshold():
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD + 0.1)])
    spotify = StubSpotify(DEFAULT_SEARCH, tracks={"t_old": _track_json("t_old", "봄날", "BTS")})
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(curator=curator, records=records, spotify=spotify).recommend(_req())

    assert res.recommendation_basis == "PERSONAL"
    assert "이 사용자가 비슷한 상황에서 들었던 곡(취향 참고용): BTS - 봄날" in curator.situations[0]


NEARBY = [
    {"title": "밤양갱", "artistName": "비비", "count": 12},
    {"title": "Hype Boy", "artistName": "NewJeans", "count": 7},
]


async def test_regional_when_no_personal_record_but_nearby_tracks():
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH)).recommend(
        _req(nearbyTracks=NEARBY)
    )

    assert res.recommendation_basis == "REGIONAL"
    nearby_line = "이 근처 사람들이 최근 자물쇠에 많이 단 곡(분위기 참고용)"
    assert f"{nearby_line}: 비비 - 밤양갱, NewJeans - Hype Boy" in curator.situations[0]


async def test_personal_wins_basis_but_nearby_tracks_still_in_prompt():
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD + 0.1)])
    spotify = StubSpotify(DEFAULT_SEARCH, tracks={"t_old": _track_json("t_old", "봄날", "BTS")})
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(curator=curator, records=records, spotify=spotify).recommend(
        _req(nearbyTracks=NEARBY)
    )

    assert res.recommendation_basis == "PERSONAL"
    assert "BTS - 봄날" in curator.situations[0]
    assert "비비 - 밤양갱" in curator.situations[0]


async def test_only_top_nearby_tracks_go_into_prompt():
    top = NEARBY_REFERENCE_COUNT
    many = [{"title": f"곡{i}", "artistName": f"가수{i}"} for i in range(top + 3)]
    curator = StubSongCurator(_curation([BALLAD]))

    await _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH)).recommend(
        _req(nearbyTracks=many)
    )

    assert f"가수{top - 1} - 곡{top - 1}" in curator.situations[0]
    assert f"가수{top} - 곡{top}" not in curator.situations[0]


async def test_past_record_below_threshold_is_generic():
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD - 0.01)])
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(
        curator=curator, records=records, spotify=StubSpotify(DEFAULT_SEARCH)
    ).recommend(_req())

    assert res.recommendation_basis == "GENERIC"


async def test_similarity_threshold_comes_from_the_caller():
    """임베딩 모델마다 점수 분포가 달라서 기준값은 설정에서 받아 온다 (모델 교체 때 같이 바뀐다)."""
    records = StubRecordSearch([SimilarRecord(1, "t_old", 0.70)])
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(
        curator=curator,
        records=records,
        spotify=StubSpotify(DEFAULT_SEARCH),
        personal_similarity=0.80,
    ).recommend(_req())

    assert res.recommendation_basis == "GENERIC"


async def test_personal_search_failure_still_recommends_as_generic():
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(
        curator=curator,
        records=StubRecordSearch(error=RuntimeError("DB 장애")),
        spotify=StubSpotify(DEFAULT_SEARCH),
    ).recommend(_req())

    assert res.recommendation_basis == "GENERIC"
    assert res.degraded is False
    assert len(res.tracks) == 1


async def test_song_already_heard_in_similar_moment_is_not_recommended_again():
    records = StubRecordSearch([SimilarRecord(1, "t_ballad", 0.9)])
    spotify = StubSpotify(
        {
            BALLAD.search_query: [_track_json("t_ballad", "밤편지", "아이유")],
            _song("잔나비", "뜨거운 여름밤").search_query: [
                _track_json("t_new", "뜨거운 여름밤", "잔나비")
            ],
        },
        tracks={"t_ballad": _track_json("t_ballad", "밤편지", "아이유")},
    )
    curator = StubSongCurator(_curation([BALLAD, _song("잔나비", "뜨거운 여름밤")]))

    res = await _service(curator=curator, records=records, spotify=spotify).recommend(_req())

    assert [t.title for t in res.tracks] == ["뜨거운 여름밤"]


async def test_tracks_carry_rerank_score_highest_first():
    exact = _song("아이유", "밤편지", genre="발라드", moods=("잔잔한", "그리운"))
    partial = _song("잔나비", "뜨거운 여름밤", genre="인디음악", moods=("잔잔한",))
    spotify = StubSpotify(
        {
            exact.search_query: [_track_json("t1", "밤편지", "아이유")],
            partial.search_query: [_track_json("t2", "뜨거운 여름밤", "잔나비")],
        }
    )
    curator = StubSongCurator(_curation([partial, exact]))

    res = await _service(curator=curator, spotify=spotify).recommend(_req())

    assert [t.title for t in res.tracks] == ["밤편지", "뜨거운 여름밤"]
    assert res.tracks[0].rerank_score > res.tracks[1].rerank_score


async def test_candidate_with_no_tag_overlap_is_dropped():
    fits = _song("아이유", "밤편지", genre="발라드", moods=("잔잔한",))
    off = _song("싸이", "강남스타일", genre="댄스", moods=("신나는",))
    spotify = StubSpotify(
        {
            fits.search_query: [_track_json("t1", "밤편지", "아이유")],
            off.search_query: [_track_json("t2", "강남스타일", "싸이")],
        }
    )
    curator = StubSongCurator(_curation([fits, off]))

    res = await _service(curator=curator, spotify=spotify).recommend(_req())

    assert [t.title for t in res.tracks] == ["밤편지"]


async def test_limit_caps_the_number_of_tracks():
    songs = [_song("A", f"Song {i}") for i in range(5)]
    spotify = StubSpotify(
        {s.search_query: [_track_json(f"t{i}", s.title, "A")] for i, s in enumerate(songs)}
    )

    res = await _service(curator=StubSongCurator(_curation(songs)), spotify=spotify).recommend(
        _req(limit=2)
    )

    assert len(res.tracks) == 2


async def test_llm_failure_is_degraded():
    res = await _service(curator=StubSongCurator(error=LLMError("장애"))).recommend(_req())

    assert res.degraded is True
    assert res.tracks == []
    assert res.request_id == "req_1"


async def test_spotify_down_for_every_song_raises_502():
    spotify = StubSpotify({BALLAD.search_query: SpotifyError("장애")})

    with pytest.raises(ExternalApiFailedError):
        await _service(curator=StubSongCurator(_curation([BALLAD])), spotify=spotify).recommend(
            _req()
        )


@pytest.mark.parametrize(
    ("hour", "bucket"),
    [(3, "새벽"), (7, "아침"), (13, "낮"), (19, "저녁"), (22, "밤")],
)
def test_time_bucket(hour, bucket):
    assert _time_bucket(datetime(2026, 8, 31, hour, 0)) == bucket


@pytest.mark.parametrize(
    ("condition", "phrase"),
    [
        ("CLEAR", "맑은"),
        ("CLOUDY", "구름 많은"),
        ("OVERCAST", "흐린"),
        ("RAIN", "비 오는"),
        ("SNOW", "눈 오는"),
        ("RAIN_SNOW", "진눈깨비 오는"),
        ("SHOWER", "소나기 오는"),
    ],
)
def test_every_backend_weather_condition_has_a_phrase(condition, phrase):
    """백엔드 WeatherCondition enum 7종 — 빠지면 영어 코드가 그대로 프롬프트에 들어간다."""
    assert _weather_phrase(condition) == phrase


async def test_place_without_name_is_left_out_of_the_prompt():
    """백엔드 places엔 법정동 이름만 있고 비어 있을 수도 있다."""
    curator = StubSongCurator(_curation([BALLAD]))

    await _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH)).recommend(
        _req(place={"placeId": 10})
    )

    assert curator.situations[0] == "상황: 비 오는 저녁, 기온 18도"


async def test_tracks_carry_spotify_track_id_and_url_for_the_backend():
    res = await _service(
        curator=StubSongCurator(_curation([BALLAD])), spotify=StubSpotify(DEFAULT_SEARCH)
    ).recommend(_req())

    assert res.tracks[0].external_track_id == "t_ballad"
    assert res.tracks[0].external_url == "https://open.spotify.com/track/t_ballad"


async def test_returns_at_most_5_even_if_backend_asks_for_10():
    """백엔드가 limit=10을 보내도 5곡까지만 — 5곡이 찼으면 LLM 보충 호출도 안 한다."""
    songs = [_song("가수", f"곡{i}") for i in range(7)]
    search = {
        s.search_query: [_track_json(f"t{i}", s.title, s.artist)] for i, s in enumerate(songs)
    }
    curator = StubSongCurator(_curation(songs))

    res = await _service(curator=curator, spotify=StubSpotify(search)).recommend(_req(limit=10))

    assert len(res.tracks) == 5
    assert curator.excludes == []


async def test_tops_up_once_when_fewer_than_5():
    first = [_song("가수", "곡0")]
    extra = [_song("다른가수", f"추가{i}") for i in range(4)]
    search = {
        s.search_query: [_track_json(f"t{i}", s.title, s.artist)] for i, s in enumerate(first)
    }
    search |= {
        s.search_query: [_track_json(f"x{i}", s.title, s.artist)] for i, s in enumerate(extra)
    }
    curator = StubSongCurator(_curation(first), extra=_curation(extra))

    res = await _service(curator=curator, spotify=StubSpotify(search)).recommend(_req(limit=5))

    assert len(res.tracks) == 5
    assert curator.excludes == [first]


# ── 곡 풀 섞기 (최대 5곡 = 곡 풀 3 + LLM 2) ─────────────────────────────


def _pool_of(*pairs, genres=("발라드",), moods=("잔잔한", "그리운")):
    import random

    from app.components.song_pool import PoolSong, SongPool

    return SongPool([PoolSong(a, t, genres, moods) for a, t in pairs], rng=random.Random(0))


def _search_for(*pairs):
    return {_song(a, t).search_query: [_track_json(f"id-{a}-{t}", t, a)] for a, t in pairs}


def _mixed_service(*, llm, pool, search, pool_count=3, records=None, tracks=None):
    curator = StubSongCurator(_curation([_song(a, t) for a, t in llm]))
    service = ContextRecommendService(
        embedder=StubEmbedder(),
        record_search=records or StubRecordSearch(),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=StubSpotify(search, tracks=tracks),
        song_pool=_pool_of(*pool),
        pool_count=pool_count,
        llm_count=4,
    )
    return service, curator


async def test_mixes_three_pool_songs_and_two_llm_songs():
    llm = [("L1", "a"), ("L2", "b"), ("L3", "c"), ("L4", "d")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z"), ("P4", "w")]
    service, curator = _mixed_service(llm=llm, pool=pool, search=_search_for(*llm, *pool))

    res = await service.recommend(_req(limit=5))

    artists = [t.artist_name for t in res.tracks]
    assert len(artists) == 5
    assert sum(a.startswith("P") for a in artists) == 3
    assert curator.counts == [4]
    # 점수 순으로 정렬돼 나온다
    scores = [t.rerank_score for t in res.tracks]
    assert scores == sorted(scores, reverse=True)


async def test_pool_does_not_repeat_songs_the_user_already_heard():
    llm = [("L1", "a"), ("L2", "b")]
    pool = [("BTS", "봄날"), ("P2", "y")]
    records = StubRecordSearch(
        [SimilarRecord(1, "id-BTS-봄날", PERSONAL_SIMILARITY_THRESHOLD + 0.1)]
    )
    tracks = {"id-BTS-봄날": _track_json("id-BTS-봄날", "봄날", "BTS")}
    service, _ = _mixed_service(
        llm=llm, pool=pool, search=_search_for(*llm, *pool), records=records, tracks=tracks
    )

    res = await service.recommend(_req(limit=5))

    assert "봄날" not in [t.title for t in res.tracks]


async def test_llm_only_when_pool_disabled():
    llm = [("L1", "a")]
    service, curator = _mixed_service(
        llm=llm, pool=[("P1", "x")], search=_search_for(*llm, ("P1", "x")), pool_count=0
    )

    res = await service.recommend(_req(limit=5))

    assert [t.artist_name for t in res.tracks] == ["L1"]
    assert curator.counts[0] == 10


async def test_respects_requested_limit_with_pool():
    llm = [("L1", "a"), ("L2", "b")]
    pool = [("P1", "x"), ("P2", "y"), ("P3", "z")]
    service, _ = _mixed_service(llm=llm, pool=pool, search=_search_for(*llm, *pool))

    res = await service.recommend(_req(limit=2))

    assert len(res.tracks) == 2


# ── 재추천 제외 (excludeTrackIds, 2026-10-16) ───────────────────────────


async def test_excluded_track_is_dropped_from_results():
    curator = StubSongCurator(_curation([BALLAD]))
    service = _service(curator=curator, spotify=StubSpotify(DEFAULT_SEARCH))

    res = await service.recommend(_req(excludeTrackIds=["t_ballad"]))

    assert res.tracks == []
    assert res.degraded is False


async def test_excluded_track_does_not_count_against_result_limit():
    songs = [BALLAD, _song("The Weeknd", "Blinding Lights")]
    search = {
        **DEFAULT_SEARCH,
        _song("The Weeknd", "Blinding Lights").search_query: [
            _track_json("t2", "Blinding Lights", "The Weeknd")
        ],
    }
    curator = StubSongCurator(_curation(songs))
    service = _service(curator=curator, spotify=StubSpotify(search))

    res = await service.recommend(_req(excludeTrackIds=["t_ballad"], limit=2))

    assert [t.external_track_id for t in res.tracks] == ["t2"]


# ── 저장 곡 + 재추천 제외가 함께 걸릴 때 (2026-10-16) ─────────────────────


def _saved_pool_of(*track_ids, genre="발라드", moods=("잔잔한", "그리운")):
    import random

    from app.components.saved_song_pool import SavedSongPool
    from app.db.repositories.track_mood_repository import SavedSong

    songs = [SavedSong(tid, genre, tuple(moods)) for tid in track_ids]

    class _Search:
        async def search_by_tags(self, genres, moods_, limit):
            return songs

    return SavedSongPool(_Search(), rng=random.Random(0))


async def test_saved_song_is_dropped_when_the_client_excluded_it():
    """재추천으로 빠져야 하는 곡이 저장 곡 쪽에서 되살아나면 안 된다."""
    tracks = {"s1": _track_json("s1", "저장곡", "가수")}
    service = ContextRecommendService(
        embedder=StubEmbedder(),
        record_search=StubRecordSearch(),
        song_curator=StubSongCurator(_curation([BALLAD])),
        reranker=Reranker(),
        spotify_search=StubSpotify(DEFAULT_SEARCH, tracks=tracks),
        saved_pool=_saved_pool_of("s1"),
        saved_count=1,
        pool_count=3,
    )

    res = await service.recommend(_req(excludeTrackIds=["s1"], limit=5))

    assert "s1" not in [t.external_track_id for t in res.tracks]
