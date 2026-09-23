import httpx

from app.clients.spotify_client import SpotifyClient


def make_client(handler) -> SpotifyClient:
    return SpotifyClient("id", "secret", timeout=5, transport=httpx.MockTransport(handler))


async def test_token_is_cached_and_track_id_is_normalized():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.host == "accounts.spotify.com":
            assert req.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert req.headers["Authorization"] == "Bearer tok"
        return httpx.Response(200, json={"id": "abc", "name": "밤편지"})

    client = make_client(handler)
    await client.get_track("spotify:track:abc")
    await client.get_track("abc")
    assert calls == ["/api/token", "/v1/tracks/abc", "/v1/tracks/abc"]


async def test_search_returns_items():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "accounts.spotify.com":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert req.url.params["type"] == "track"
        return httpx.Response(200, json={"tracks": {"items": [{"id": "1"}, {"id": "2"}]}})

    assert [t["id"] for t in await make_client(handler).search_tracks("밤편지")] == ["1", "2"]
