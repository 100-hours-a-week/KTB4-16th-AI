from app.clients.spotify_client import SpotifyError
from app.components.song_curator import SongCandidate
from app.components.track_resolver import TrackResolver


class StubSpotify:
    def __init__(self, results: dict[str, list[dict] | Exception]):
        self._results = results
        self.queries: list[str] = []

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        self.queries.append(query)
        result = self._results.get(query, [])
        if isinstance(result, Exception):
            raise result
        return result


def _track(track_id: str, title: str, *artists: str) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": a} for a in artists],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": []},
    }


def _song(artist: str, title: str) -> SongCandidate:
    return SongCandidate(artist=artist, title=title, genre="발라드", moods=("잔잔한",))


async def test_exact_field_filter_hit_skips_fallback_search():
    song = _song("아이유", "밤편지")
    spotify = StubSpotify({song.search_query: [_track("t1", "밤편지", "아이유")]})

    resolved = await TrackResolver(spotify).resolve([song])

    assert [c.external_track_id for c in resolved] == ["t1"]
    assert spotify.queries == [song.search_query]


async def test_spacing_difference_is_rescued_by_fallback():
    """실측: LLM "아닐 거야" vs Spotify "아닐거야" — 필드 필터는 0건을 준다."""
    song = _song("백예린", "그건 아마 우리의 잘못은 아닐 거야")
    spotify = StubSpotify(
        {
            f"{song.title} {song.artist}": [
                _track(
                    "karaoke",
                    "Maybe It's Not Our Fault (그건 아마 우리의 잘못은 아닐 거야)",
                    "ZZang KARAOKE",
                ),
                _track("t1", "그건 아마 우리의 잘못은 아닐거야", "백예린"),
            ]
        }
    )

    resolved = await TrackResolver(spotify).resolve([song])

    assert [c.external_track_id for c in resolved] == ["t1"]


async def test_fallback_rejects_same_title_by_another_artist():
    """실측: "윤딴딴 - Bad Habits"를 그냥 검색하면 Ed Sheeran 곡이 1위로 나온다."""
    song = _song("윤딴딴", "Bad Habits")
    spotify = StubSpotify(
        {f"{song.title} {song.artist}": [_track("t1", "Bad Habits", "Ed Sheeran")]}
    )

    assert await TrackResolver(spotify).resolve([song]) == []


async def test_fallback_rejects_different_title():
    """실측: "정승환 - 노을"을 그냥 검색하면 "노라조 - 형"이 1위로 나온다."""
    song = _song("정승환", "노을")
    spotify = StubSpotify({f"{song.title} {song.artist}": [_track("t1", "형", "노라조")]})

    assert await TrackResolver(spotify).resolve([song]) == []


async def test_fallback_ignores_parenthesized_and_remaster_suffixes():
    songs = [_song("백예린", "Square"), _song("The Beatles", "Here Comes The Sun")]
    spotify = StubSpotify(
        {
            "Square 백예린": [_track("t1", "Square (2017)", "백예린")],
            "Here Comes The Sun The Beatles": [
                _track("t2", "Here Comes The Sun - Remastered 2009", "The Beatles")
            ],
        }
    )

    resolved = await TrackResolver(spotify).resolve(songs)

    assert [c.external_track_id for c in resolved] == ["t1", "t2"]


async def test_artist_matches_any_of_the_featured_artists():
    song = _song("아이유", "동화")
    spotify = StubSpotify(
        {"동화 아이유": [_track("t1", "동화 (feat. 아이유)", "김동률", "아이유")]}
    )

    assert [c.external_track_id for c in await TrackResolver(spotify).resolve([song])] == ["t1"]


async def test_one_failing_song_does_not_drop_the_others():
    ok, broken = _song("아이유", "밤편지"), _song("장애", "곡")
    spotify = StubSpotify(
        {
            ok.search_query: [_track("t1", "밤편지", "아이유")],
            broken.search_query: SpotifyError("일시 장애"),
        }
    )

    assert [c.external_track_id for c in await TrackResolver(spotify).resolve([ok, broken])] == [
        "t1"
    ]


async def test_prefers_exact_title_over_a_medley_that_contains_it():
    """실측: 필드 필터 1위가 제목에 그 단어가 들어간 라이브 메들리였다."""
    song = _song("Earth, Wind & Fire", "Saturday Nite")
    spotify = StubSpotify(
        {
            song.search_query: [
                _track(
                    "medley",
                    "Serpentine Fire / Saturday Nite / Reasons - Live",
                    "Earth, Wind & Fire",
                ),
                _track("t1", "Saturday Nite", "Earth, Wind & Fire"),
            ]
        }
    )

    assert [c.external_track_id for c in await TrackResolver(spotify).resolve([song])] == ["t1"]


async def test_external_url_comes_from_spotify_or_is_built_from_the_id():
    with_url, without_url = _song("아이유", "밤편지"), _song("잔나비", "주저하는 연인들을 위해")
    track_with_url = {
        **_track("t1", "밤편지", "아이유"),
        "external_urls": {"spotify": "https://open.spotify.com/track/t1?si=abc"},
    }
    spotify = StubSpotify(
        {
            with_url.search_query: [track_with_url],
            without_url.search_query: [_track("t2", "주저하는 연인들을 위해", "잔나비")],
        }
    )

    resolved = await TrackResolver(spotify).resolve([with_url, without_url])

    assert [c.external_url for c in resolved] == [
        "https://open.spotify.com/track/t1?si=abc",
        "https://open.spotify.com/track/t2",
    ]
