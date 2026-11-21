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
        # 순차 호출 — 병렬로 여러 개를 한꺼번에 보내면 계정 요청 제한(429)에
        # 걸리기 쉽다(실측: 태그 14개 동시 호출 시 429 다발). 카테고리 벡터
        # 캐싱처럼 여러 텍스트를 한 번에 인코딩할 때가 유일한 다건 호출 지점.
        return [await self._predict({"text": t}) for t in texts]

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


class RunPodClipClient:
    """RunPod Pod에서 직접 띄운 CLIP 서버(scripts/clip_server/clip_server.py)를 부른다.

    Replicate openai/clip과 같은 모델(clip-vit-large-patch14, 768차원)이라 저장된 벡터와 비교할 수
    있다 (2026-11-06 같은 사진 41장 비교: 이미지 벡터 코사인 중앙값 1.0000).
    입력: POST /encode_image {"image": url} → {"embedding": [...]},
          POST /encode_text {"texts": [...]} → {"embeddings": [[...], ...]}
    사진 주소는 Replicate처럼 서버가 직접 내려받는다.
    """

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    RETRY_WAIT_SECONDS = 2.0
    # 서버가 한 번에 받는 글 개수 (clip_server.py의 MAX_TEXTS)
    MAX_TEXTS_PER_REQUEST = 64
    # RunPod 앞단이 파이썬 기본 User-Agent를 막은 적이 있어서(403 error code 1010)
    # 직접 정한 값을 보낸다
    USER_AGENT = "muro-ai-clip/1.0"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout: float,
        max_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        # model은 서버의 모델 이름이 아니라 서비스가 저장·비교에 쓰는 이름이다.
        # 태그 벡터 버전(tag_version)과 저장된 벡터의 모델 표시가 이 값으로 정해지므로
        # Replicate 때와 같은 값(CLIP_MODEL=openai/clip)을 그대로 쓴다
        self.model = model
        self.dim = dim
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": self.USER_AGENT},
            timeout=timeout,
            transport=transport,
        )

    async def encode_image(self, image_url: str) -> list[float]:
        body = await self._post("/encode_image", {"image": image_url})
        return self._check_dim(body.get("embedding"))

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_TEXTS_PER_REQUEST):
            batch = texts[start : start + self.MAX_TEXTS_PER_REQUEST]
            body = await self._post("/encode_text", {"texts": batch})
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ClipError("CLIP 응답의 embeddings 개수가 보낸 글 개수와 달라요.")
            vectors.extend(self._check_dim(e) for e in embeddings)
        return vectors

    def _check_dim(self, embedding: Any) -> list[float]:
        if not embedding:
            raise ClipError("CLIP 응답에 embedding이 없어요.")
        if len(embedding) != self.dim:
            raise ClipError(f"CLIP 차원({len(embedding)})이 설정값({self.dim})과 달라요.")
        return embedding

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 연결 실패와 5xx는 Pod 재시작 중일 수 있어서 다시 보낸다.
        # 4xx(키 틀림, 사진 못 받음 등)는 바로 실패
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                res = await self._http.post(url, json=payload)
            except httpx.HTTPError as e:
                if last:
                    raise ClipError(f"CLIP 서버 연결 실패: {type(e).__name__}") from e
                await asyncio.sleep(self.RETRY_WAIT_SECONDS * (attempt + 1))
                continue
            if res.status_code in self.RETRYABLE_STATUS and not last:
                await asyncio.sleep(self.RETRY_WAIT_SECONDS * (attempt + 1))
                continue
            if res.status_code >= 400:
                raise ClipError(f"CLIP 서버 오류 (status {res.status_code})")
            return res.json()
        raise ClipError("CLIP 서버 재시도 초과")  # 위 반복에서 항상 반환하거나 예외를 낸다


class FakeClipClient:
    def __init__(self, dim: int, model: str = "fake-clip"):
        self.model = model
        self.dim = dim

    async def encode_image(self, image_url: str) -> list[float]:
        return fake_vector(f"image:{image_url}", self.dim)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"clip-text:{t}", self.dim) for t in texts]
