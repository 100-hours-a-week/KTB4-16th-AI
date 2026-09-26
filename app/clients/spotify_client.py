"""Spotify Web API 어댑터.

두 계정 체계를 쓴다 (기능1 플레이리스트 저장 — B안: 서비스 계정 + 팔로우 링크).
- SpotifyClient: 앱 인증(Client Credentials) — 검색·곡·아티스트 조회. 사용자 계정 불필요, 무제한
- SpotifyServiceAccountClient: 뮤로 팀 계정 1개의 refresh token으로 그 계정 안에 플레이리스트 생성
  (사용자 개별 OAuth 없음 — Development Mode 25명 제한과 무관)
"""

import asyncio
import base64
import time
from typing import Any, Protocol

import httpx

from app.exceptions import UpstreamError


class SpotifyError(UpstreamError):
    message = "Spotify API 호출에 실패했어요."


class SpotifySearchClient(Protocol):
    """검색·곡·아티스트 조회 — SpotifyClient가 실제 구현체."""

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict[str, Any]]: ...
    async def get_track(self, track_id: str) -> dict[str, Any]: ...
    async def get_artist(self, artist_id: str) -> dict[str, Any]: ...


class SpotifyPlaylistClient(Protocol):
    """플레이리스트 저장 (B안) — SpotifyServiceAccountClient가 실제 구현체."""

    async def create_playlist_with_tracks(
        self, *, title: str, track_uris: list[str], description: str = ""
    ) -> str: ...


class SpotifyClient:
    API_URL = "https://api.spotify.com/v1"
    TOKEN_URL = "https://accounts.spotify.com/api/token"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        timeout: float,
        market: str = "KR",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._market = market
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        data = await self._get(
            "/search", params={"q": query, "type": "track", "limit": limit, "market": self._market}
        )
        return data.get("tracks", {}).get("items", [])

    async def get_track(self, track_id: str) -> dict[str, Any]:
        return await self._get(f"/tracks/{_to_id(track_id)}", params={"market": self._market})

    async def get_artist(self, artist_id: str) -> dict[str, Any]:
        return await self._get(f"/artists/{artist_id}")

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        token = await self._access_token()
        try:
            res = await self._http.get(
                f"{self.API_URL}{path}",
                params=params,
                # 없으면 곡·아티스트명이 영어로 옴 ("밤편지" → "Through the Night")
                headers={"Authorization": f"Bearer {token}", "Accept-Language": "ko"},
            )
        except httpx.HTTPError as e:
            raise SpotifyError(f"Spotify 연결 실패: {type(e).__name__}") from e
        if res.status_code >= 400:
            raise SpotifyError(f"Spotify API 오류 (status {res.status_code}): {res.text[:200]}")
        return res.json()

    async def _access_token(self) -> str:
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at - 60:
                return self._token
            basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
            try:
                res = await self._http.post(
                    self.TOKEN_URL,
                    data={"grant_type": "client_credentials"},
                    headers={"Authorization": f"Basic {basic}"},
                )
            except httpx.HTTPError as e:
                raise SpotifyError(f"Spotify 토큰 발급 연결 실패: {type(e).__name__}") from e
            if res.status_code >= 400:
                raise SpotifyError(f"Spotify 토큰 발급 실패 (status {res.status_code})")
            body = res.json()
            self._token = body["access_token"]
            self._token_expires_at = time.monotonic() + body.get("expires_in", 3600)
            return self._token


class SpotifyServiceAccountClient:
    """뮤로 서비스 계정(1개) 안에 추천 플레이리스트를 만들어 공유 링크를 돌려준다.

    refresh_token은 scripts/spotify_service_account_setup.py로 한 번만 발급받아 .env에 저장.
    """

    API_URL = "https://api.spotify.com/v1"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    # 플레이리스트 생성·수정에 필요한 최소 권한
    SCOPES = "playlist-modify-public playlist-modify-private"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._service_user_id: str | None = None

    async def create_playlist_with_tracks(
        self, *, title: str, track_uris: list[str], description: str = ""
    ) -> str:
        """플레이리스트를 만들고 곡을 채운 뒤, 사용자에게 보여줄 공유 링크를 돌려준다."""
        user_id = await self._service_user()
        playlist = await self._request(
            "POST",
            f"/users/{user_id}/playlists",
            json={"name": title, "description": description, "public": False},
        )
        if track_uris:
            await self._request(
                "POST",
                f"/playlists/{playlist['id']}/tracks",
                json={"uris": track_uris},
            )
        return playlist["external_urls"]["spotify"]

    async def _service_user(self) -> str:
        if self._service_user_id is None:
            me = await self._request("GET", "/me")
            self._service_user_id = me["id"]
        return self._service_user_id

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        token = await self._access_token()
        try:
            res = await self._http.request(
                method,
                f"{self.API_URL}{path}",
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
        except httpx.HTTPError as e:
            raise SpotifyError(f"Spotify 연결 실패: {type(e).__name__}") from e
        if res.status_code >= 400:
            raise SpotifyError(f"Spotify API 오류 (status {res.status_code}): {res.text[:200]}")
        return res.json() if res.content else {}

    async def _access_token(self) -> str:
        """refresh_token으로 access_token 재발급 (사용자 재로그인 없이 계속 갱신됨)."""
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at - 60:
                return self._token
            basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
            try:
                res = await self._http.post(
                    self.TOKEN_URL,
                    data={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
                    headers={"Authorization": f"Basic {basic}"},
                )
            except httpx.HTTPError as e:
                raise SpotifyError(f"Spotify 토큰 갱신 연결 실패: {type(e).__name__}") from e
            if res.status_code >= 400:
                raise SpotifyError(
                    f"Spotify 토큰 갱신 실패 (status {res.status_code}) — refresh_token 재발급 필요"
                )
            body = res.json()
            self._token = body["access_token"]
            self._token_expires_at = time.monotonic() + body.get("expires_in", 3600)
            return self._token


def _to_id(track_id: str) -> str:
    """'spotify:track:xxx' 형태도 받아서 ID만 남긴다."""
    return track_id.rsplit(":", 1)[-1]


class FakeSpotifySearchClient:
    async def search_tracks(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "id": f"fake_{i}",
                "uri": f"spotify:track:fake_{i}",
                "name": f"{query} 결과 {i}",
                "artists": [{"name": "Fake Artist", "id": "fake_artist"}],
                "album": {"name": "Fake Album", "images": [], "release_date": "2024-01-01"},
            }
            for i in range(min(limit, 3))
        ]

    async def get_track(self, track_id: str) -> dict[str, Any]:
        return {
            "id": _to_id(track_id),
            "name": "Fake Track",
            "artists": [{"name": "Fake Artist", "id": "fake_artist"}],
            "album": {"name": "Fake Album", "images": [], "release_date": "2024-01-01"},
        }

    async def get_artist(self, artist_id: str) -> dict[str, Any]:
        return {"id": artist_id, "name": "Fake Artist", "genres": []}


class FakeSpotifyPlaylistClient:
    async def create_playlist_with_tracks(
        self, *, title: str, track_uris: list[str], description: str = ""
    ) -> str:
        return f"https://open.spotify.com/playlist/fake_{abs(hash(title)) % 100000}"
