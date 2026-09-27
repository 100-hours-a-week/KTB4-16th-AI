from fastapi import APIRouter, Depends, status

from app.db.repositories.job_repository import JobRepository
from app.dependencies import get_job_repository
from app.schemas.reports import ReportBatchGenerateRequest, ReportBatchQueuedResponse
from app.services.report_service import ReportService

router = APIRouter(prefix="/api", tags=["기능2 RECAP"])


@router.post(
    "/reports/batch-generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReportBatchQueuedResponse,
)
async def batch_generate(
    req: ReportBatchGenerateRequest,
    jobs: JobRepository = Depends(get_job_repository),
) -> ReportBatchQueuedResponse:
    return await ReportService(jobs).enqueue_batch(req)
