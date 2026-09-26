import httpx

from app.clients.spotify_client import SpotifyClient, SpotifyServiceAccountClient


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


def make_service_client(handler) -> SpotifyServiceAccountClient:
    return SpotifyServiceAccountClient(
        "id", "secret", "refresh-tok", timeout=5, transport=httpx.MockTransport(handler)
    )


async def test_create_playlist_uses_refresh_token_and_returns_share_url():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.url.host == "accounts.spotify.com":
            assert "refresh_token=refresh-tok" in req.content.decode()
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if req.url.path == "/v1/me":
            return httpx.Response(200, json={"id": "muro_official"})
        if req.url.path == "/v1/users/muro_official/playlists":
            assert req.headers["Authorization"] == "Bearer tok"
            return httpx.Response(
                201,
                json={
                    "id": "pl1",
                    "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"},
                },
            )
        if req.url.path == "/v1/playlists/pl1/tracks":
            return httpx.Response(201, json={"snapshot_id": "abc"})
        raise AssertionError(f"unexpected call {req.method} {req.url.path}")

    client = make_service_client(handler)
    url = await client.create_playlist_with_tracks(
        title="비 오는 저녁의 뮤로", track_uris=["spotify:track:1", "spotify:track:2"]
    )
    assert url == "https://open.spotify.com/playlist/pl1"
    assert ("GET", "/v1/me") in calls
    assert ("POST", "/v1/playlists/pl1/tracks") in calls


async def test_create_playlist_without_tracks_skips_add_call():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "accounts.spotify.com":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if req.url.path == "/v1/me":
            return httpx.Response(200, json={"id": "muro_official"})
        if req.url.path == "/v1/users/muro_official/playlists":
            return httpx.Response(
                201, json={"id": "pl1", "external_urls": {"spotify": "https://x/pl1"}}
            )
        raise AssertionError("곡이 없으면 /tracks 호출하면 안 됨")

    url = await make_service_client(handler).create_playlist_with_tracks(
        title="빈 목록", track_uris=[]
    )
    assert url == "https://x/pl1"


async def test_service_account_reuses_service_user_id():
    me_calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "accounts.spotify.com":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        if req.url.path == "/v1/me":
            me_calls.append(1)
            return httpx.Response(200, json={"id": "muro_official"})
        if "playlists" in req.url.path and req.method == "POST" and "tracks" not in req.url.path:
            return httpx.Response(201, json={"id": "pl", "external_urls": {"spotify": "https://x"}})
        return httpx.Response(201, json={})

    client = make_service_client(handler)
    await client.create_playlist_with_tracks(title="A", track_uris=[])
    await client.create_playlist_with_tracks(title="B", track_uris=[])
    assert len(me_calls) == 1
