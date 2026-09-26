from app.clients.spotify_client import FakeSpotifyPlaylistClient
from app.exceptions import NotImplementedFeatureError
from app.schemas.context_recommend import PlaylistSaveRequest
from app.services.playlist_service import PlaylistService


async def test_save_spotify_returns_share_url():
    req = PlaylistSaveRequest(
        userId=1044,
        trackUris=["spotify:track:aaa", "spotify:track:bbb"],
        platform="spotify",
        title="비 오는 저녁의 뮤로",
    )
    res = await PlaylistService(FakeSpotifyPlaylistClient()).save(req)
    assert res.platform == "spotify"
    assert res.playlist_url.startswith("https://open.spotify.com/playlist/")


async def test_save_youtube_not_implemented_yet():
    req = PlaylistSaveRequest(
        userId=1, trackUris=["spotify:track:aaa"], platform="youtube", title="t"
    )
    try:
        await PlaylistService(FakeSpotifyPlaylistClient()).save(req)
        raise AssertionError("expected NotImplementedFeatureError")
    except NotImplementedFeatureError:
        pass
