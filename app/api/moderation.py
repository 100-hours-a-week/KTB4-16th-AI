import time

from fastapi import APIRouter, Depends

from app.dependencies import get_moderation_model, get_moderation_service
from app.schemas.health import ModerationHealthResponse
from app.schemas.moderation import ModerationCheckRequest, ModerationCheckResponse
from app.services.moderation_service import ModerationService

# 인증이 필요한 라우터와 헬스체크 라우터를 분리한다
router = APIRouter(prefix="/api/moderation", tags=["기능6 모더레이션"])
health_router = APIRouter(prefix="/api/moderation", tags=["health"])

_started_at = time.monotonic()


@router.post("/check", response_model=ModerationCheckResponse)
async def check(
    req: ModerationCheckRequest,
    service: ModerationService = Depends(get_moderation_service),
) -> ModerationCheckResponse:
    return await service.check(req)


@health_router.get("/health", response_model=ModerationHealthResponse)
async def moderation_health() -> ModerationHealthResponse:
    # 모델 파일이 없으면 modelVersion은 null (서버는 뜨고 판정만 503)
    model = get_moderation_model()
    return ModerationHealthResponse(
        status="ok",
        model_version=model.version if model else None,
        uptime_seconds=int(time.monotonic() - _started_at),
    )
