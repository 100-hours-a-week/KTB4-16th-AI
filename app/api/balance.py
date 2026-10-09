from fastapi import APIRouter, Depends

from app.dependencies import get_balance_service
from app.schemas.balance import QuestionGenerateRequest, QuestionGenerateResponse
from app.services.balance_service import BalanceService

router = APIRouter(prefix="/api/balance", tags=["기능6 취향 투표"])


@router.post("/questions/generate", response_model=QuestionGenerateResponse)
async def generate_questions(
    req: QuestionGenerateRequest,
    service: BalanceService = Depends(get_balance_service),
) -> QuestionGenerateResponse:
    return await service.generate_questions(req)
