from fastapi import APIRouter

from app.schemas.context_recommend import PlaylistSaveRequest, PlaylistSaveResponse
from app.services.playlist_service import PlaylistService

router = APIRouter(prefix="/api", tags=["기능1 상황 맞춤 추천"])


@router.post("/playlist/save", response_model=PlaylistSaveResponse)
async def save_playlist(req: PlaylistSaveRequest) -> PlaylistSaveResponse:
    return await PlaylistService().save(req)
