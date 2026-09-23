from fastapi import APIRouter

from app.schemas.context_recommend import ContextRecommendRequest, ContextRecommendResponse
from app.services.context_recommend_service import ContextRecommendService

router = APIRouter(prefix="/api", tags=["기능1 상황 맞춤 추천"])


@router.post("/context-recommend", response_model=ContextRecommendResponse)
async def context_recommend(req: ContextRecommendRequest) -> ContextRecommendResponse:
    return await ContextRecommendService().recommend(req)
