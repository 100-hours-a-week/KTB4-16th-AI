from fastapi import APIRouter

from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse
from app.services.photo_recommend_service import PhotoRecommendService

router = APIRouter(prefix="/api", tags=["기능3 사진 기반 추천"])


@router.post("/photo-recommend", response_model=PhotoRecommendResponse)
async def photo_recommend(req: PhotoRecommendRequest) -> PhotoRecommendResponse:
    return await PhotoRecommendService().recommend(req)
