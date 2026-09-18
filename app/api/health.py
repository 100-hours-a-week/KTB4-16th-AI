import time

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])

_started_at = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", uptime_seconds=int(time.monotonic() - _started_at))
