"""기능2 RECAP 배치 — 위키 모델 API 설계 4절."""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel


class ReportBatchGenerateRequest(CamelModel):
    year: int = Field(ge=2026)
    month: int = Field(ge=1, le=12)
    # 생략 시 전체 사용자. 백엔드 명세상 userId는 숫자(Long)
    user_ids: list[int] | None = None
    # 백엔드가 요청마다 주는 배치 구분값(같은 달 수동 트리거를 2번 보내도 구분하려고).
    # 콜백에 값 그대로 돌려준다. 생략하면(구버전 백엔드) 연·월로 구분하던 기존 동작.
    # ':'를 못 쓰게 막아 job 키 "{id}:{userId}"의 경계가 모호해지지 않게 한다.
    batch_request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{1,64}$")


class ReportBatchQueuedResponse(CamelModel):
    status: Literal["QUEUED"] = "QUEUED"
    job_id: str
    target_count: int | None = None
    batch_request_id: str | None = None


class AiRecapOut(CamelModel):
    # V1은 저장된 행이 있으면 항상 완료 상태 — 실패한 배치는 행 자체가 없다
    status: Literal["COMPLETED"] = "COMPLETED"
    text: str


class PhotoSceneOut(CamelModel):
    tag: str
    count: int
    ratio: int


class MonthlyReportDetailResponse(CamelModel):
    """기분·아티스트·장소 통계는 안 담는다 — 백엔드가 자기 MySQL로 직접 계산하기로 협의됨."""

    user_id: int
    year: int
    month: int
    photo_scenes: list[PhotoSceneOut]
    ai_recap: AiRecapOut
