"""상태 확인 — 위키 모델 API 설계 9절."""

from app.schemas.common import CamelModel


class HealthResponse(CamelModel):
    status: str
    uptime_seconds: int


class ModerationHealthResponse(HealthResponse):
    model_version: str | None
