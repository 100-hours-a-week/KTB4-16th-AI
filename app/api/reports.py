from fastapi import APIRouter, status

from app.schemas.reports import ReportBatchGenerateRequest, ReportBatchQueuedResponse
from app.services.report_service import ReportService

router = APIRouter(prefix="/api", tags=["기능2 RECAP"])


@router.post(
    "/reports/batch-generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ReportBatchQueuedResponse,
)
async def batch_generate(req: ReportBatchGenerateRequest) -> ReportBatchQueuedResponse:
    return await ReportService().enqueue_batch(req)
