"""사진 주소(백엔드가 주는 서명된 GCS 주소)에서 사진을 내려받아 data: URL로 만든다 — 기능3.

사진을 모델 서버(RunPod)가 직접 받게 하지 않는 이유: 서명 주소가 모델 서버에서 열리는지 보장할 수
없고, 우리 쪽에서 크기·종류를 먼저 확인할 수 있다. 리다이렉트는 따라가지 않는다(주소가 다른 곳으로
넘어가 내부 주소를 읽는 것을 막는다).
"""

import base64
import logging

import httpx

logger = logging.getLogger("muro.image_fetcher")

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif")


class ImageFetchError(Exception):
    """사진을 받지 못했다 (주소 문제·크기·종류·네트워크)."""


class ImageFetcher:
    def __init__(self, timeout: float = 20.0, transport: httpx.AsyncBaseTransport | None = None):
        self._timeout = timeout
        self._transport = transport

    async def fetch_data_url(self, url: str) -> str:
        if not url.lower().startswith(("https://", "http://")):
            raise ImageFetchError("사진 주소는 http(s)만 받아요.")
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport, follow_redirects=False
            ) as client:
                res = await client.get(url)
            res.raise_for_status()
        except httpx.HTTPError as e:
            raise ImageFetchError(f"사진을 받지 못했어요 ({type(e).__name__})") from e
        content_type = res.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_TYPES:
            raise ImageFetchError(
                f"지원하지 않는 사진 형식이에요 ({content_type or '알 수 없음'})."
            )
        if len(res.content) > MAX_BYTES:
            raise ImageFetchError("사진이 너무 커요.")
        if not res.content:
            raise ImageFetchError("사진이 비어 있어요.")
        return f"data:{content_type};base64,{base64.b64encode(res.content).decode()}"
