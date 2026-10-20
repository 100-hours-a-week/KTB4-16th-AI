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
    def __init__(
        self,
        hits: list[SimilarRecord] | None = None,
        error: Exception | None = None,
        taste: list[list[float]] | None = None,
        taste_error: Exception | None = None,
        rainy: dict | None = None,
    ):
        self._rainy = rainy or {}
        self._hits = hits or []
        self._error = error
        self._taste = taste or []
        self._taste_error = taste_error
        self.calls: list[int] = []

    async def similar_by_comment(self, user_id, query_vector, limit):
        self.calls.append(user_id)
        if self._error:
            raise self._error
        return self._hits

    async def music_vectors(self, user_id, limit):
        if self._taste_error:
            raise self._taste_error
        return self._taste

    async def music_vectors_for_records(self, user_id, record_ids):
        self.asked_records = record_ids
        return self._rainy.get("vectors", [])


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


class _VectorRecordingPool:
    """곡 풀 대역 — 곡을 고를 때 받은 질문 벡터를 기록한다."""

    def __len__(self):
        return 1

    def ranked(self, genres, moods, exclude=None, situation_vector=None):
        self.vector = situation_vector
        return []


def _tagged(sentence="비 오는 날 조용한 노래"):
    return CurationResult(("발라드",), ("잔잔한",), [BALLAD], mood_sentence=sentence)


async def _picked_vector(*, records, taste_weight):
    pool = _VectorRecordingPool()
    service = ContextRecommendService(
        embedder=StubEmbedder(),  # 상황 문장 → [1, 0, 0]
        record_search=records,
        song_curator=StubSongCurator(_tagged()),
        reranker=Reranker(),
        spotify_search=StubSpotify(DEFAULT_SEARCH),
        song_pool=pool,
        pool_count=3,
        taste_weight=taste_weight,
    )
    await service.recommend(_req())
    return pool.vector


async def test_taste_vector_is_blended_into_the_pool_query_by_weight():
    records = StubRecordSearch(taste=[[0.0, 1.0, 0.0]])

    off = await _picked_vector(records=records, taste_weight=0.0)
    on = await _picked_vector(records=records, taste_weight=0.5)

    assert off == [1.0, 0.0, 0.0]
    assert on[0] > 0 and on[1] > 0  # 상황과 취향이 섞였다


async def test_taste_failure_or_no_history_falls_back_to_the_situation_vector():
    broken = StubRecordSearch(taste_error=RuntimeError("DB 장애"))
    empty = StubRecordSearch(taste=[])

    assert await _picked_vector(records=broken, taste_weight=0.3) == [1.0, 0.0, 0.0]
    assert await _picked_vector(records=empty, taste_weight=0.3) == [1.0, 0.0, 0.0]


class _Rows:
    """백엔드 자물쇠 정보 대역 — (record_id, 날씨) 목록."""

    def __init__(self, rows):
        from datetime import datetime

        from app.db.repositories.memory_record_repository import ConditionRow

        self._rows = [
            ConditionRow(rid, datetime(2026, 10, 1), w, None, None, None, None, None, None)
            for rid, w in rows
        ]

    async def existing_ids(self, *, user_id, record_ids):
        return set()

    async def list_condition_rows(self, *, user_id, limit):
        return self._rows


async def _vector_with(*, records, record_source, nearby=None, pool=None, nearby_weight=0.0):
    pool = pool or _VectorRecordingPool()
    service = ContextRecommendService(
        embedder=StubEmbedder(),
        record_search=records,
        song_curator=StubSongCurator(_tagged()),
        reranker=Reranker(),
        spotify_search=StubSpotify(DEFAULT_SEARCH),
        song_pool=pool,
        pool_count=3,
        taste_weight=0.5,
        record_source=record_source,
        situational_min_records=2,
        nearby_weight=nearby_weight,
    )
    extra = {"nearbyTracks": nearby} if nearby else {}
    await service.recommend(_req(**extra))
    return pool.vector


async def test_taste_uses_only_records_saved_in_the_same_weather_when_enough():
    # 지금 날씨는 RAIN(요청 기본값). 맑은 날 기록(3)은 빼고 비 오는 날 기록(1, 2)만 쓴다
    records = StubRecordSearch(taste=[[0.0, 0.0, 1.0]], rainy={"vectors": [[0.0, 1.0, 0.0]] * 2})
    source = _Rows([(1, "RAIN"), (2, "rain"), (3, "CLEAR")])

    vector = await _vector_with(records=records, record_source=source)

    assert sorted(records.asked_records) == [1, 2]
    assert vector[1] > 0 and vector[2] == 0  # 전체 평균(축 3)이 아니라 비 오는 날 기록(축 2)


async def test_taste_falls_back_to_overall_when_too_few_same_weather_records():
    records = StubRecordSearch(taste=[[0.0, 0.0, 1.0]], rainy={"vectors": [[0.0, 1.0, 0.0]]})
    source = _Rows([(1, "RAIN"), (3, "CLEAR")])  # 비 오는 날 기록 1개 < 기준 2개

    vector = await _vector_with(records=records, record_source=source)

    assert vector[2] > 0 and vector[1] == 0  # 전체 평균 쪽


def _pool_with_vectors():
    import random

    from app.components.song_pool import PoolSong, SongPool

    songs = [
        PoolSong("아이유", "밤편지", ("발라드",), ("잔잔한",), 0, (0.0, 1.0, 0.0)),
        PoolSong("볼빨간사춘기", "나의 사춘기에게", ("발라드",), ("잔잔한",), 0, (0.0, 1.0, 0.0)),
    ]
    return SongPool(songs, rng=random.Random(0))


class _VectorRecordingRealPool:
    """실제 곡 풀을 감싸서 pick에 들어온 질문 벡터를 기록한다."""

    def __init__(self, pool):
        self._pool = pool

    def __len__(self):
        return len(self._pool)

    def vectors_for(self, tracks):
        return self._pool.vectors_for(tracks)

    def ranked(self, *args, situation_vector=None, **kwargs):
        self.vector = situation_vector
        return []


async def test_nearby_songs_in_the_pool_pull_the_query_toward_them():
    pool = _VectorRecordingRealPool(_pool_with_vectors())
    nearby = [
        {"title": "밤편지", "artistName": "아이유", "count": 5},
        {"title": "풀에 없는 곡", "artistName": "모르는 가수", "count": 9},
    ]

    off = await _vector_with(
        records=StubRecordSearch(), record_source=None, nearby=nearby, pool=pool
    )
    on = await _vector_with(
        records=StubRecordSearch(), record_source=None, nearby=nearby, pool=pool, nearby_weight=0.5
    )

    assert off[1] == 0 and on[1] > 0  # 곡 풀에 있는 곡 쪽(축 2)으로 이동


async def test_db_songs_come_first_in_hybrid_score_order_and_the_claude_song_comes_last():
    import random

    from app.components.song_pool import PoolSong, SongPool

    pool = SongPool(
        [
            PoolSong("A", "먼곡", ("발라드",), ("잔잔한", "그리운"), 0, (0.0, 1.0, 0.0)),
            PoolSong("B", "가까운곡", ("발라드",), ("잔잔한", "그리운"), 0, (1.0, 0.0, 0.0)),
        ],
        rng=random.Random(0),
    )
    claude = _song("L", "클로드곡", moods=("잔잔한", "그리운"))
    curation = CurationResult(
        ("발라드",), ("잔잔한", "그리운"), [claude], mood_sentence="비 오는 저녁의 쓸쓸한 노래"
    )
    service = ContextRecommendService(
        embedder=StubEmbedder(),  # 상황 문장 벡터 [1, 0, 0] — B와 같은 방향
        record_search=StubRecordSearch(),
        song_curator=StubSongCurator(curation),
        reranker=Reranker(),
        spotify_search=StubSpotify(
            _search_for(("L", "클로드곡"), ("A", "먼곡"), ("B", "가까운곡"))
        ),
        song_pool=pool,
        pool_count=2,
        llm_count=3,
    )

    res = await service.recommend(_req(limit=5))

    assert [t.title for t in res.tracks] == ["가까운곡", "먼곡", "클로드곡"]
    assert res.tracks[0].rerank_score == 1.0 and res.tracks[1].rerank_score == 0.5


async def test_saved_and_pool_songs_compete_for_the_db_slots_in_the_context_flow():
    import random

    from app.components.song_pool import PoolSong, SongPool
    from app.db.repositories.track_mood_repository import SavedSong

    pool = SongPool(
        [
            PoolSong(f"가수{i}", f"풀곡{i}", ("발라드",), ("잔잔한", "그리운"), 0, (0.0, 1.0, 0.0))
            for i in range(4)
        ],
        rng=random.Random(0),
    )
    close_saved = SavedSong("s_close", "발라드", ("잔잔한", "그리운"), embedding=(1.0, 0.0, 0.0))
    saved = _saved_pool_with(close_saved)
    curation = CurationResult(("발라드",), ("잔잔한", "그리운"), [], mood_sentence="비 오는 저녁")
    tracks = {"s_close": _track_json("s_close", "저장곡", "누군가")}
    search = _search_for(*[(f"가수{i}", f"풀곡{i}") for i in range(4)])
    service = ContextRecommendService(
        embedder=StubEmbedder(),
        record_search=StubRecordSearch(),
        song_curator=StubSongCurator(curation),
        reranker=Reranker(),
        spotify_search=StubSpotify(search, tracks=tracks),
        song_pool=pool,
        pool_count=3,
        saved_pool=saved,
        saved_count=1,
    )

    res = await service.recommend(_req(limit=5))

    assert res.tracks[0].external_track_id == "s_close"  # 분위기가 가장 가까운 저장된 곡이 1등
    assert len(res.tracks) == 3  # DB 몫 3곡


def _saved_pool_with(*songs):
    import random

    from app.components.saved_song_pool import SavedSongPool

    class _Search:
        async def search_by_tags(self, genres, moods_, limit):
            return list(songs)

    return SavedSongPool(_Search(), rng=random.Random(0))


class _HangingEmbedder:
    model = "hang"

    async def embed(self, text):
        import asyncio

        await asyncio.sleep(30)


async def test_slow_embedding_is_skipped_so_the_request_is_not_held(monkeypatch):
    import time

    from app.services import context_recommend_service as svc

    monkeypatch.setattr(svc, "EMBED_TIMEOUT_SECONDS", 0.05)
    pool = _VectorRecordingPool()
    service = ContextRecommendService(
        embedder=_HangingEmbedder(),
        record_search=StubRecordSearch(),
        song_curator=StubSongCurator(_tagged()),
        reranker=Reranker(),
        spotify_search=StubSpotify(DEFAULT_SEARCH),
        song_pool=pool,
        pool_count=3,
    )

    started = time.monotonic()
    await service.recommend(_req())

    assert time.monotonic() - started < 5  # 30초 걸리는 임베딩을 기다리지 않는다
    assert pool.vector is None  # 벡터 없이 태그 점수로 계속했다


class _SlowRecordSearch(StubRecordSearch):
    async def music_vectors(self, user_id, limit):
        import asyncio

        await asyncio.sleep(30)


async def test_slow_taste_lookup_is_skipped_and_the_situation_vector_is_used(monkeypatch):
    import time

    from app.services import context_recommend_service as svc

    monkeypatch.setattr(svc, "TASTE_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()

    vector = await _picked_vector(records=_SlowRecordSearch(), taste_weight=0.3)

    assert time.monotonic() - started < 5
    assert vector == [1.0, 0.0, 0.0]  # 취향을 못 읽어서 상황 벡터 그대로
