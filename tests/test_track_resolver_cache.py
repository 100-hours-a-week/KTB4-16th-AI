"""TrackResolver의 Spotify 확인 결과 캐시(track_lookups) 동작."""

from datetime import UTC, datetime, timedelta

from app.components.song_curator import SongCandidate
from app.components.track_resolver import TrackResolver
from app.db.repositories.track_lookup_repository import LookupEntry, SpotifyTrack
from tests.conftest import InMemoryLookups

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)


class CountingSpotify:
    def __init__(self, results: dict[str, list[dict]]):
        self.results = results
        self.queries: list[str] = []

    async def search_tracks(self, query, limit=10):
        self.queries.append(query)
        return self.results.get(query, [])


def _song(artist="아이유", title="밤편지") -> SongCandidate:
    return SongCandidate(artist=artist, title=title, genre="발라드", moods=("잔잔한",))


def _json(track_id="t1", title="밤편지", artist="아이유") -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": artist}],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": [{"url": f"https://img/{track_id}.jpg"}]},
    }


def _entry(track: SpotifyTrack | None, age: timedelta, key="아이유|밤편지") -> LookupEntry:
    return LookupEntry(
        lookup_key=key,
        llm_artist="아이유",
        llm_title="밤편지",
        genre="발라드",
        moods=("잔잔한",),
        track=track,
        verified_at=NOW - age,
    )


TRACK = SpotifyTrack("t1", "밤편지", "아이유", "spotify:track:t1", "https://img/t1.jpg", "u")


async def test_first_lookup_queries_spotify_and_saves():
    store = InMemoryLookups()
    spotify = CountingSpotify({_song().search_query: [_json()]})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert [t.external_track_id for t in tracks] == ["t1"]
    saved = store.rows["아이유|밤편지"]
    assert saved.track.external_track_id == "t1"
    assert saved.llm_title == "밤편지" and saved.genre == "발라드"


async def test_cached_song_skips_spotify():
    store = InMemoryLookups([_entry(TRACK, timedelta(days=3))])
    spotify = CountingSpotify({})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert spotify.queries == []
    assert [t.external_track_id for t in tracks] == ["t1"]


async def test_spacing_difference_hits_same_cache_entry():
    store = InMemoryLookups([_entry(TRACK, timedelta(days=1))])
    spotify = CountingSpotify({})

    await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song(title="밤 편지")])

    assert spotify.queries == []


async def test_expired_entry_is_checked_again():
    store = InMemoryLookups([_entry(TRACK, timedelta(days=31))])
    spotify = CountingSpotify({_song().search_query: [_json()]})

    await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert spotify.queries == [_song().search_query]
    assert store.rows["아이유|밤편지"].verified_at == NOW


async def test_not_found_is_remembered_for_a_day():
    """LLM이 지어낸 곡은 하루 동안 다시 검색하지 않는다."""
    store = InMemoryLookups([_entry(None, timedelta(hours=5))])
    spotify = CountingSpotify({})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert tracks == []
    assert spotify.queries == []


async def test_not_found_is_checked_again_after_a_day():
    store = InMemoryLookups([_entry(None, timedelta(days=2))])
    spotify = CountingSpotify({_song().search_query: [_json()]})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert [t.external_track_id for t in tracks] == ["t1"]


async def test_cache_failure_falls_back_to_spotify():
    store = InMemoryLookups(broken=True)
    spotify = CountingSpotify({_song().search_query: [_json()]})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song()])

    assert [t.external_track_id for t in tracks] == ["t1"]


async def test_cached_track_uses_this_requests_tags():
    """재랭킹 기준을 맞추려고 장르·무드는 저장값이 아니라 이번 LLM 태그를 쓴다."""
    store = InMemoryLookups([_entry(TRACK, timedelta(days=1))])
    song = SongCandidate(artist="아이유", title="밤편지", genre="인디", moods=("그리운",))

    tracks = await TrackResolver(CountingSpotify({}), store, now=lambda: NOW).resolve([song])

    assert tracks[0].genre == "인디" and tracks[0].moods == ("그리운",)


async def test_duplicate_song_is_searched_and_saved_once():
    store = InMemoryLookups()
    spotify = CountingSpotify({_song().search_query: [_json()]})

    tracks = await TrackResolver(spotify, store, now=lambda: NOW).resolve([_song(), _song()])

    assert spotify.queries == [_song().search_query]
    assert [t.external_track_id for t in tracks] == ["t1"]
