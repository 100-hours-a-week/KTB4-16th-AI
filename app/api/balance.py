from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status

from app.db.repositories.job_repository import JobRepository
from app.dependencies import get_balance_service, get_job_repository
from app.schemas.balance import (
    DailyBatchQueuedResponse,
    DailyBatchRequest,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
)
from app.services.balance_service import BalanceService
from app.workers import balance_daily_worker

router = APIRouter(prefix="/api/balance", tags=["기능6 취향 투표"])


@router.post("/questions/generate", response_model=QuestionGenerateResponse)
async def generate_questions(
    req: QuestionGenerateRequest,
    service: BalanceService = Depends(get_balance_service),
) -> QuestionGenerateResponse:
    return await service.generate_questions(req)


@router.post(
    "/daily-batch", status_code=status.HTTP_202_ACCEPTED, response_model=DailyBatchQueuedResponse
)
async def run_daily_batch(
    req: DailyBatchRequest, jobs: JobRepository = Depends(get_job_repository)
) -> DailyBatchQueuedResponse:
    """운영자용 수동 실행(다시 돌리기 포함) — 평소에는 워커가 매일 23:00(KST)에 알아서 넣는다."""
    target = req.date or balance_daily_worker.tomorrow(datetime.now(UTC))
    await jobs.enqueue(**balance_daily_worker.job_fields(target))
    return DailyBatchQueuedResponse(date=target)
