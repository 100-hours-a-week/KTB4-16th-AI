"""Spotify Web API 어댑터 — 앱 인증(Client Credentials)으로 검색·곡·아티스트 조회.

사용자 계정이 필요한 기능(플레이리스트 저장)은 이 인증으로 할 수 없다.
"""

import asyncio
import base64
import time
from typing import Any

import httpx

from app.exceptions import UpstreamError


class SpotifyError(UpstreamError):
    message = "Spotify API 호출에 실패했어요."


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


def _to_id(track_id: str) -> str:
    """'spotify:track:xxx' 형태도 받아서 ID만 남긴다."""
    return track_id.rsplit(":", 1)[-1]
