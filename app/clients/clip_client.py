"""CLIP API 어댑터 — 이미지·텍스트를 같은 CLIP 공간으로 인코딩."""

import asyncio
from typing import Any, Protocol

import httpx

from app.clients.fake import fake_vector
from app.exceptions import ClipError


class ClipClient(Protocol):
    model: str
    dim: int

    async def encode_image(self, image_url: str) -> list[float]: ...

    async def encode_text(self, texts: list[str]) -> list[list[float]]: ...


class ReplicateClipClient:
    """Replicate의 openai/clip (clip-vit-large-patch14, 768차원).

    입력: {"image": url} 또는 {"text": str} / 출력: {"embedding": [...]}
    이미지 벡터와 라벨(텍스트) 벡터가 같은 모델에서 나오므로 서로 비교할 수 있다.
    """

    BASE_URL = "https://api.replicate.com/v1"
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    POLL_INTERVAL_SECONDS = 1.0
    # 429는 계정 호출 제한이라 잠깐 기다리면 풀린다 — 일반 오류보다 넉넉히 재시도
    RATE_LIMIT_EXTRA_RETRIES = 6
    RATE_LIMIT_WAIT_SECONDS = 5.0

    def __init__(
        self,
        api_token: str,
        model: str,
        dim: int,
        timeout: float,
        max_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.dim = dim
        self._timeout = timeout
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=timeout,
            transport=transport,
        )

    async def encode_image(self, image_url: str) -> list[float]:
        return await self._predict({"image": image_url})

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        # 모델이 한 번에 한 입력만 받으므로 병렬 호출
        return list(await asyncio.gather(*(self._predict({"text": t}) for t in texts)))

    async def _predict(self, model_input: dict[str, str]) -> list[float]:
        prediction = await self._create_prediction(model_input)
        prediction = await self._wait_until_done(prediction)

        if prediction.get("status") != "succeeded":
            raise ClipError(f"CLIP 예측 실패: {prediction.get('status')} {prediction.get('error')}")
        embedding = (prediction.get("output") or {}).get("embedding")
        if not embedding:
            raise ClipError("CLIP 응답에 embedding이 없어요.")
        if len(embedding) != self.dim:
            raise ClipError(f"CLIP 차원({len(embedding)})이 설정값({self.dim})과 달라요.")
        return embedding

    async def _create_prediction(self, model_input: dict[str, str]) -> dict[str, Any]:
        wait = max(1, min(60, int(self._timeout)))
        return await self._request(
            "POST",
            f"/models/{self.model}/predictions",
            json={"input": model_input},
            headers={"Prefer": f"wait={wait}"},
        )

    async def _wait_until_done(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """sync 대기 시간 안에 안 끝났으면 완료될 때까지 조회한다."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        while prediction.get("status") in ("starting", "processing"):
            if loop.time() > deadline:
                raise ClipError("CLIP 예측 대기 시간 초과")
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
            prediction = await self._request("GET", f"/predictions/{prediction['id']}")
        return prediction

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        max_attempts = self._max_retries + 1 + self.RATE_LIMIT_EXTRA_RETRIES
        errors, rate_limited = 0, 0
        for _ in range(max_attempts):
            try:
                res = await self._http.request(method, url, **kwargs)
            except httpx.HTTPError as e:
                errors += 1
                if errors <= self._max_retries:
                    await asyncio.sleep(2**errors)
                    continue
                raise ClipError(f"CLIP API 연결 실패: {type(e).__name__}") from e

            if res.status_code == 429:
                rate_limited += 1
                if rate_limited <= self.RATE_LIMIT_EXTRA_RETRIES:
                    await asyncio.sleep(self._retry_after(res))
                    continue
            elif res.status_code in self.RETRYABLE_STATUS:
                errors += 1
                if errors <= self._max_retries:
                    await asyncio.sleep(2**errors)
                    continue
            if res.status_code >= 400:
                raise ClipError(f"CLIP API 오류 (status {res.status_code})")
            return res.json()
        raise ClipError("CLIP API 재시도 초과 (호출 제한)")

    def _retry_after(self, res: httpx.Response) -> float:
        """Replicate가 알려준 대기 시간을 우선 따른다."""
        header = res.headers.get("retry-after")
        if header:
            try:
                return min(60.0, max(1.0, float(header)))
            except ValueError:
                pass
        return self.RATE_LIMIT_WAIT_SECONDS


class FakeClipClient:
    def __init__(self, dim: int, model: str = "fake-clip"):
        self.model = model
        self.dim = dim

    async def encode_image(self, image_url: str) -> list[float]:
        return fake_vector(f"image:{image_url}", self.dim)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"clip-text:{t}", self.dim) for t in texts]
