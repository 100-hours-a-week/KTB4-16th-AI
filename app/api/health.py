import time

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])

_started_at = time.monotonic()


# /health는 배포 스모크 테스트(CD)가 확인하는 주소. 둘 다 같은 응답
@router.get("/api/health", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", uptime_seconds=int(time.monotonic() - _started_at))
