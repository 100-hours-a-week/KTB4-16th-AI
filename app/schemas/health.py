"""상태 확인 — 위키 모델 API 설계 9절."""

from app.schemas.common import CamelModel


class HealthResponse(CamelModel):
    status: str
    uptime_seconds: int


class ModerationHealthResponse(HealthResponse):
    model_version: str | None


class EmbeddingHealthResponse(HealthResponse):
    # 모델 파일이 없으면 model은 null (서버는 뜨고 임베딩 요청만 503)
    model: str | None
    dim: int | None
