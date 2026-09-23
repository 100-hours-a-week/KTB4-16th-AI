"""기능2 RECAP 배치 — 위키 모델 API 설계 4절."""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel


class ReportBatchGenerateRequest(CamelModel):
    year: int = Field(ge=2026)
    month: int = Field(ge=1, le=12)
    # 생략 시 전체 사용자
    user_ids: list[str] | None = None


class ReportBatchQueuedResponse(CamelModel):
    status: Literal["QUEUED"] = "QUEUED"
    job_id: str
    target_count: int | None = None
