"""게이트웨이(8000)에서 모더레이션 서비스(8001)로 요청을 그대로 넘기는 경로.

모더레이션은 채팅 검열이 일반 API에 밀려 느려지지 않게 프로세스(포트 8001)를 따로 둔다.
그런데 8001을 밖(백엔드)에 열려면 인스턴스 템플릿·내부 로드밸런서·방화벽을 모두 바꿔야 해서,
이미 열려 있는 8000에서 같은 경로(/api/moderation/check)를 받아 컨테이너 안의 8001로
전달한다. 백엔드는 AI base URL 하나로 모더레이션까지 부른다 (주소 8000 + /api/moderation/check).

요청·응답은 가공하지 않고 그대로 주고받는다 — 판정은 모더레이션 서비스가 하고, 여기는 전달만 한다.
모더레이션이 돌려주는 오류(모델 없음 503 등)도 그대로 내려가고, 연결에 실패하면 503
UPSTREAM_UNAVAILABLE을 준다 (백엔드는 실패를 "차단"으로 처리한다).
"""

import logging
from functools import lru_cache

import httpx
from fastapi import APIRouter, Depends, Request, Response

from app.config import get_settings
from app.exceptions import UpstreamError

logger = logging.getLogger("muro.moderation_proxy")

router = APIRouter(prefix="/api/moderation", tags=["기능6 모더레이션"])
health_router = APIRouter(prefix="/api/moderation", tags=["health"])

CHECK_PATH = "/api/moderation/check"
HEALTH_PATH = "/api/moderation/health"


@lru_cache
def get_moderation_client() -> httpx.AsyncClient:
    s = get_settings()
    return httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{s.moderation_port}",
        timeout=s.moderation_proxy_timeout_seconds,
    )


async def _forward(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> Response:
    try:
        res = await client.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise UpstreamError("모더레이션 서비스에 연결하지 못했어요.") from e
    return Response(
        content=res.content,
        status_code=res.status_code,
        media_type=res.headers.get("content-type", "application/json"),
    )


@router.post("/check")
async def check(
    request: Request, client: httpx.AsyncClient = Depends(get_moderation_client)
) -> Response:
    body = await request.body()
    if not body:
        # 호출하는 쪽이 JSON 헤더만 붙이고 본문을 안 보낸 경우 — 안쪽에서 "body: Field required"로만
        # 보여서 원인을 찾기 어려웠다. 어디서 비어 왔는지 여기서 남긴다 (본문 내용은 남기지 않는다)
        logger.warning(
            "모더레이션 요청 본문이 비어 있음 (content-length=%s, content-type=%s) — "
            "전달하다 빠진 게 아니라 받을 때부터 비어 있었음",
            request.headers.get("content-length"),
            request.headers.get("content-type"),
        )
    return await _forward(
        client,
        "POST",
        CHECK_PATH,
        content=body,
        headers={
            "Content-Type": request.headers.get("content-type", "application/json"),
            # 모더레이션 서비스도 같은 토큰을 확인한다
            "X-Internal-Token": get_settings().internal_token,
        },
    )


@health_router.get("/health")
async def health(client: httpx.AsyncClient = Depends(get_moderation_client)) -> Response:
    return await _forward(client, "GET", HEALTH_PATH)
