"""CLIP API 어댑터 — 이미지·텍스트를 같은 CLIP 공간으로 인코딩."""

from typing import Protocol

from app.clients.fake import fake_vector


class ClipClient(Protocol):
    model: str
    dim: int

    async def encode_image(self, image_url: str) -> list[float]: ...

    async def encode_text(self, texts: list[str]) -> list[list[float]]: ...


class HttpClipClient:
    """CLIP 제공처 확정 후 구현.

    이미지 벡터와 라벨(텍스트) 벡터는 반드시 같은 모델로 만들어야 비교가 성립한다.
    실패는 ClipError로 변환해 올린다.
    """

    def __init__(self, api_url: str, api_key: str, model: str, dim: int, timeout: float):
        self.model = model
        self.dim = dim
        self._api_url = api_url
        self._api_key = api_key
        self._timeout = timeout

    async def encode_image(self, image_url: str) -> list[float]:
        raise NotImplementedError("CLIP 제공처 확정 후 구현 (AI_CLIENT_MODE=fake로 우회 가능)")

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("CLIP 제공처 확정 후 구현 (AI_CLIENT_MODE=fake로 우회 가능)")


class FakeClipClient:
    def __init__(self, dim: int, model: str = "fake-clip"):
        self.model = model
        self.dim = dim

    async def encode_image(self, image_url: str) -> list[float]:
        return fake_vector(f"image:{image_url}", self.dim)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"clip-text:{t}", self.dim) for t in texts]
