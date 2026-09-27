from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifyPlaylistClient
from app.dependencies import get_spotify_playlist_client
from app.schemas.context_recommend import PlaylistSaveRequest, PlaylistSaveResponse
from app.services.playlist_service import PlaylistService

router = APIRouter(prefix="/api", tags=["기능1 상황 맞춤 추천"])


@router.post("/playlist/save", response_model=PlaylistSaveResponse)
async def save_playlist(
    req: PlaylistSaveRequest,
    spotify_playlist: SpotifyPlaylistClient = Depends(get_spotify_playlist_client),
) -> PlaylistSaveResponse:
    return await PlaylistService(spotify_playlist).save(req)
