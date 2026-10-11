from fastapi import APIRouter, Depends, status

from app.db.repositories.job_repository import JobRepository
from app.db.repositories.report_repository import MonthlyReportRepository
from app.dependencies import get_job_repository, get_monthly_report_repository
from app.exceptions import MonthlyReportNotFoundError
from app.schemas.reports import (
    AiRecapOut,
    MonthlyReportDetailResponse,
    ReportBatchGenerateRequest,
    ReportBatchQueuedResponse,
)
from app.services.report_service import ReportService

router = APIRouter(prefix="/api", tags=["기능2 RECAP"])


@router.post(
    "/reports/batch-generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReportBatchQueuedResponse,
    # batchRequestId 없이 온 요청엔 응답에도 그 필드를 안 넣는다 — 기존 응답 모양 그대로
    response_model_exclude_none=True,
)
async def batch_generate(
    req: ReportBatchGenerateRequest,
    jobs: JobRepository = Depends(get_job_repository),
) -> ReportBatchQueuedResponse:
    return await ReportService(jobs).enqueue_batch(req)


@router.get("/reports/{user_id}", response_model=MonthlyReportDetailResponse)
async def get_monthly_report(
    user_id: int,
    year: int,
    month: int,
    reports: MonthlyReportRepository = Depends(get_monthly_report_repository),
) -> MonthlyReportDetailResponse:
    """배치 완료 콜백을 받은 백엔드가 실제 리포트 내용을 가져갈 때 호출한다."""
    record = await reports.get(user_id=user_id, year=year, month=month)
    if record is None:
        raise MonthlyReportNotFoundError()
    return MonthlyReportDetailResponse(
        user_id=user_id,
        year=year,
        month=month,
        photo_scenes=record.photo_scenes,
        ai_recap=AiRecapOut(text=record.ai_recap_text),
    )
