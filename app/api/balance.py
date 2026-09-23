from fastapi import APIRouter, status

from app.schemas.balance import (
    ClusterAssignRequest,
    ClusterAssignResponse,
    ClusterRebuildQueuedResponse,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
)
from app.services.balance_service import BalanceService

router = APIRouter(prefix="/api/balance", tags=["기능6 취향 투표"])


@router.post(
    "/clusters/rebuild",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ClusterRebuildQueuedResponse,
)
async def rebuild_clusters() -> ClusterRebuildQueuedResponse:
    return await BalanceService().handle()


@router.post("/questions/generate", response_model=QuestionGenerateResponse)
async def generate_questions(req: QuestionGenerateRequest) -> QuestionGenerateResponse:
    return await BalanceService().handle(req)


@router.post("/cluster-assign", response_model=ClusterAssignResponse)
async def assign_cluster(req: ClusterAssignRequest) -> ClusterAssignResponse:
    return await BalanceService().handle(req)
