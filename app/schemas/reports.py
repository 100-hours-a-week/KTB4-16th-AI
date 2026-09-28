"""기능2 RECAP 배치 — 위키 모델 API 설계 4절."""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel


class ReportBatchGenerateRequest(CamelModel):
    year: int = Field(ge=2026)
    month: int = Field(ge=1, le=12)
    # 생략 시 전체 사용자. 백엔드 명세상 userId는 숫자(Long)
    user_ids: list[int] | None = None


class ReportBatchQueuedResponse(CamelModel):
    status: Literal["QUEUED"] = "QUEUED"
    job_id: str
    target_count: int | None = None


class AiRecapOut(CamelModel):
    # V1은 저장된 행이 있으면 항상 완료 상태 — 실패한 배치는 행 자체가 없다
    status: Literal["COMPLETED"] = "COMPLETED"
    text: str


class MonthlyReportDetailResponse(CamelModel):
    """기분·아티스트·장소 통계는 안 담는다 — 백엔드가 자기 MySQL로 직접 계산하기로 협의됨."""

    user_id: int
    year: int
    month: int
    photo_scenes: list[str]
    ai_recap: AiRecapOut
