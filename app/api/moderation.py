import time

from fastapi import APIRouter

from app.schemas.health import ModerationHealthResponse
from app.schemas.moderation import ModerationCheckRequest, ModerationCheckResponse
from app.services.moderation_service import ModerationService

# 인증이 필요한 라우터와 헬스체크 라우터를 분리한다
router = APIRouter(prefix="/api/moderation", tags=["기능6 모더레이션"])
health_router = APIRouter(prefix="/api/moderation", tags=["health"])

_started_at = time.monotonic()


@router.post("/check", response_model=ModerationCheckResponse)
async def check(req: ModerationCheckRequest) -> ModerationCheckResponse:
    return await ModerationService().check(req)


@health_router.get("/health", response_model=ModerationHealthResponse)
async def moderation_health() -> ModerationHealthResponse:
    # 모델 로드 전이라 modelVersion은 null
    return ModerationHealthResponse(
        status="ok", model_version=None, uptime_seconds=int(time.monotonic() - _started_at)
    )
