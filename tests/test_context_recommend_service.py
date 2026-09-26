from datetime import datetime

import pytest

from app.clients.spotify_client import SpotifyError
from app.components.reranker import Reranker
from app.components.song_curator import CurationResult, SongCandidate
from app.components.vector_search import SimilarRecord
from app.exceptions import LLMError
from app.schemas.context_recommend import ContextRecommendRequest
from app.services.context_recommend_service import (
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
    def __init__(self, curation: CurationResult | None = None, error: Exception | None = None):
        self._curation = curation
        self._error = error
        self.situations: list[str] = []

    async def curate(self, situation: str) -> CurationResult:
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
        "album": {"images": []},
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


def _service(*, curator, spotify=None, records=None, embedder=None):
    return ContextRecommendService(
        embedder=embedder or StubEmbedder(),
        record_search=records or StubRecordSearch(),
        song_curator=curator,
        reranker=Reranker(),
        spotify_search=spotify or StubSpotify(),
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


async def test_past_record_below_threshold_is_generic():
    records = StubRecordSearch([SimilarRecord(1, "t_old", PERSONAL_SIMILARITY_THRESHOLD - 0.01)])
    curator = StubSongCurator(_curation([BALLAD]))

    res = await _service(
        curator=curator, records=records, spotify=StubSpotify(DEFAULT_SEARCH)
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
    partial = _song("잔나비", "뜨거운 여름밤", genre="인디", moods=("잔잔한",))
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


async def test_spotify_down_for_every_song_is_degraded():
    spotify = StubSpotify({BALLAD.search_query: SpotifyError("장애")})

    res = await _service(curator=StubSongCurator(_curation([BALLAD])), spotify=spotify).recommend(
        _req()
    )

    assert res.degraded is True


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
