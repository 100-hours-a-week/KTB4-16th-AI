import random

import pytest

from app.clients.spotify_client import SpotifyError
from app.components.scene_curator import Scene
from app.components.vector_search import SimilarRecord
from app.exceptions import ExternalApiFailedError, LLMError
from app.schemas.context_recommend import ContextRecommendRequest
from app.services.context_recommend_service import (
    PERSONAL_SIMILARITY_THRESHOLD,
    ContextRecommendService,
    _situation_text,
    _time_bucket,
    _weather_phrase,
)
from tests.scene_fakes import (
    SCENES,
    StubEmbedder,
    StubSceneCurator,
    StubSpotify,
    big_pool,
    pool_of,
    song,
)


class StubRecordSearch:
    def __init__(self, hits=None, error=None, genres=None, genres_error=None):
        self._hits = hits or []
        self._error = error
        self._genres = genres or {}
        self._genres_error = genres_error

    async def similar_by_comment(self, user_id, query_vector, limit):
        if self._error:
            raise self._error
        return self._hits

    async def genres_by_record(self, user_id):
        if self._genres_error:
            raise self._genres_error
        return self._genres


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


def _service(*, pool, curator=None, spotify=None, records=None, embedder=None, **extra):
    songs = pool._songs if pool is not None else []
    return ContextRecommendService(
        embedder=embedder or StubEmbedder(),
        record_search=records or StubRecordSearch(),
        scene_curator=curator or StubSceneCurator(),
        spotify_search=spotify or StubSpotify(songs),
        song_pool=pool,
        rng=random.Random(0),
        **extra,
    )


def _small_pool():
    songs = [
        song("아이유", "밤편지", vector=(1.0, 0.0, 0.0)),
        song("폴킴", "비", vector=(0.9, 0.1, 0.0)),
        song("김광석", "사랑했지만", vector=(0.8, 0.2, 0.0)),
        song("버스커버스커", "벚꽃 엔딩", "인디음악", vector=(0.0, 1.0, 0.0)),
    ]
    return pool_of(songs)


def test_situation_text_has_season_weekday_time_weather_temperature_and_place():
    assert _situation_text(_req()) == (
        "여름(8월), 평일 저녁, 비 오는 날 (기온 18도), 장소: 동탄역 카페거리"
    )


def test_situation_text_marks_weekend_and_winter_and_skips_missing_parts():
    req = _req(
        place=None,
        weather={"condition": "SNOW"},
        localTime="2026-12-05T08:10:00+09:00",  # 토요일
    )

    assert _situation_text(req) == "겨울(12월), 주말 아침, 눈 오는 날"


def test_time_bucket_and_weather_phrase():
    assert _time_bucket(_req(localTime="2026-08-31T03:00:00+09:00").local_time) == "새벽"
    assert _weather_phrase("rain") == "비 오는"
    assert _weather_phrase("HAIL") == "HAIL"


async def test_curator_gets_the_situation_text():
    curator = StubSceneCurator()

    await _service(pool=_small_pool(), curator=curator).recommend(_req())

    assert curator.situations == [
        "여름(8월), 평일 저녁, 비 오는 날 (기온 18도), 장소: 동탄역 카페거리"
    ]


async def test_every_scene_gets_two_sentences_embedded_in_one_batch():
    embedder = StubEmbedder()

    await _service(pool=_small_pool(), embedder=embedder).recommend(_req())

    lyrics = [s.lyric for s in SCENES]
    descriptions = [s.description for s in SCENES]
    assert embedder.batches == [lyrics + descriptions]


async def test_personal_search_uses_only_weather_and_time_bucket():
    """장소 이름은 과거 코멘트에 거의 안 나와 유사도만 흐려서 검색 질의에서 뺀다."""
    embedder = StubEmbedder()

    await _service(pool=_small_pool(), embedder=embedder).recommend(_req())

    assert embedder.texts == ["비 오는 저녁"]


async def test_recommends_best_matching_songs_from_the_pool():
    res = await _service(pool=_small_pool()).recommend(_req(limit=3))

    assert res.degraded is False
    assert res.recommendation_basis == "GENERIC"
    assert {t.title for t in res.tracks} == {"밤편지", "비", "사랑했지만"}
    assert res.rewritten_query.startswith("여름(8월)")


async def test_recommend_reason_is_built_from_the_situation_and_the_picked_genres():
    """장면 문장이 아니라 상황 값(날씨·계절·평일/주말·시간대)과 고른 곡의 장르로 만든다."""
    res = await _service(pool=_small_pool()).recommend(_req(limit=3))

    assert res.recommend_reason == "비 오는 여름 평일 저녁이라 발라드 위주로 골라봤어요."
    assert all(s.scene not in res.recommend_reason for s in SCENES)


async def test_recommend_reason_mentions_taste_only_when_the_taste_filter_was_used():
    pool, songs = big_pool()
    records = StubRecordSearch(genres={1: "발라드", 2: "발라드"})

    async def onboarding(user_id):
        return ["댄스"]

    used = await _service(
        pool=pool, records=records, onboarding=onboarding, spotify=StubSpotify(songs)
    ).recommend(_req(limit=5))

    async def rare(user_id):
        return ["트로트"]  # 곡풀에 트로트 곡이 없어서 거르지 못한다

    widened = await _service(pool=pool, onboarding=rare, spotify=StubSpotify(songs)).recommend(
        _req(limit=5)
    )

    assert used.recommend_reason.endswith("좋아하시는 장르(발라드·댄스)도 반영했어요.")
    assert "좋아하시는" not in widened.recommend_reason


async def test_unknown_weather_is_left_out_of_the_reason():
    res = await _service(pool=_small_pool()).recommend(_req(weather={"condition": "HAIL"}))

    assert res.recommend_reason.startswith("여름 평일 저녁이라 ")


async def test_same_artist_appears_once():
    pool = pool_of(
        [
            song("아이유", "밤편지", vector=(1.0, 0.0, 0.0)),
            song("아이유", "Blueming", vector=(0.99, 0.01, 0.0)),
            song("폴킴", "비", vector=(0.9, 0.1, 0.0)),
            song("김광석", "사랑했지만", vector=(0.8, 0.2, 0.0)),
        ]
    )

    res = await _service(pool=pool).recommend(_req(limit=3))

    assert [t.artist_name for t in res.tracks].count("아이유") == 1
    assert len(res.tracks) == 3


async def test_personal_when_similar_past_record_passes_threshold_and_heard_song_is_skipped():
    pool = _small_pool()
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD + 0.1)])
    spotify = StubSpotify(pool._songs)

    class OneHeard(StubSpotify):
        async def get_track(self, track_id):
            return {
                "id": track_id,
                "name": "밤편지",
                "artists": [{"name": "아이유"}],
                "uri": "spotify:track:t_old",
                "album": {"images": [{"url": "x"}]},
            }

    spotify = OneHeard(pool._songs)

    res = await _service(pool=pool, records=records, spotify=spotify).recommend(_req())

    assert res.recommendation_basis == "PERSONAL"
    assert "밤편지" not in [t.title for t in res.tracks]


async def test_past_record_below_threshold_is_generic():
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD - 0.01)])

    res = await _service(pool=_small_pool(), records=records).recommend(_req())

    assert res.recommendation_basis == "GENERIC"


async def test_regional_when_no_personal_record_but_nearby_tracks():
    res = await _service(pool=_small_pool()).recommend(
        _req(nearbyTracks=[{"title": "밤양갱", "artistName": "비비", "count": 12}])
    )

    assert res.recommendation_basis == "REGIONAL"


async def test_personal_record_search_failure_falls_back_to_generic():
    records = StubRecordSearch(error=RuntimeError("db down"))

    res = await _service(pool=_small_pool(), records=records).recommend(_req())

    assert res.recommendation_basis == "GENERIC"
    assert res.tracks


async def test_exclude_track_ids_are_removed_and_replaced_by_extra_picks():
    pool = _small_pool()
    first = await _service(pool=pool).recommend(_req(limit=2))
    shown = [t.external_track_id for t in first.tracks]

    second = await _service(pool=pool).recommend(_req(limit=2, excludeTrackIds=shown))

    assert len(second.tracks) == 2
    assert not set(shown) & {t.external_track_id for t in second.tracks}


async def test_limit_is_capped_at_five():
    pool, _ = big_pool()

    res = await _service(pool=pool).recommend(_req(limit=10))

    assert len(res.tracks) == 5


async def test_scene_failure_returns_degraded():
    curator = StubSceneCurator(error=LLMError("down"))

    res = await _service(pool=_small_pool(), curator=curator).recommend(_req())

    assert res.degraded is True
    assert res.tracks == []
    assert res.recommend_reason is None


async def test_embedding_failure_returns_degraded():
    res = await _service(
        pool=_small_pool(), embedder=StubEmbedder(error=RuntimeError("embed down"))
    ).recommend(_req())

    assert res.degraded is True


async def test_missing_pool_returns_degraded():
    res = await _service(pool=None).recommend(_req())

    assert res.degraded is True


async def test_vector_length_mismatch_returns_degraded():
    res = await _service(pool=_small_pool(), embedder=StubEmbedder(vector=(1.0, 0.0))).recommend(
        _req()
    )

    assert res.degraded is True


async def test_spotify_down_raises_external_api_failed():
    pool = _small_pool()
    spotify = StubSpotify(pool._songs, error=SpotifyError("Spotify API 오류 (status 429)"))

    with pytest.raises(ExternalApiFailedError):
        await _service(pool=pool, spotify=spotify).recommend(_req())


async def test_songs_missing_on_spotify_are_dropped():
    pool = _small_pool()
    spotify = StubSpotify([s for s in pool._songs if s.title != "밤편지"])

    res = await _service(pool=pool, spotify=spotify).recommend(_req(limit=3))

    assert "밤편지" not in [t.title for t in res.tracks]


async def test_taste_genres_filter_the_candidates_and_one_slot_is_outside():
    """자물쇠 곡 장르(발라드)와 온보딩 장르(댄스)를 합쳐 거르고, 한 자리는 취향 밖(록)."""
    pool, songs = big_pool()
    records = StubRecordSearch(genres={1: "발라드", 2: "발라드"})

    async def onboarding(user_id):
        return ["댄스"]

    res = await _service(
        pool=pool, records=records, onboarding=onboarding, spotify=StubSpotify(songs)
    ).recommend(_req(limit=5))

    genre_of = {s.title: s.genres[0] for s in songs}
    genres = [genre_of[t.title] for t in res.tracks]
    assert len(genres) == 5
    assert genres.count("록/메탈") == 1
    assert genres[4] == "록/메탈"  # 취향 밖 곡은 마지막 자리(limit - 1)
    assert set(genres[:4]) <= {"발라드", "댄스"}


async def test_no_taste_means_no_filter_and_no_outside_slot():
    pool, songs = big_pool()

    res = await _service(pool=pool, spotify=StubSpotify(songs)).recommend(_req(limit=5))

    genre_of = {s.title: s.genres[0] for s in songs}
    assert {genre_of[t.title] for t in res.tracks} == {"발라드"}  # 점수 상위 20곡이 모두 발라드


async def test_taste_read_failure_does_not_fail_the_recommendation():
    pool, songs = big_pool()
    records = StubRecordSearch(genres_error=RuntimeError("pg down"))

    async def onboarding(user_id):
        raise RuntimeError("mysql down")

    res = await _service(
        pool=pool, records=records, onboarding=onboarding, spotify=StubSpotify(songs)
    ).recommend(_req(limit=5))

    assert len(res.tracks) == 5


def test_scene_dataclass_is_used_for_the_reason():
    assert isinstance(SCENES[0], Scene)
