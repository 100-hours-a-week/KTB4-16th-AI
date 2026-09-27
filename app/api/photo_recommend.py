from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.reranker import Reranker
from app.components.song_curator import SongCurator
from app.dependencies import (
    get_clip_tagger,
    get_reranker,
    get_song_curator,
    get_spotify_search_client,
)
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse
from app.services.photo_recommend_service import PhotoRecommendService

router = APIRouter(prefix="/api", tags=["기능3 사진 기반 추천"])


@router.post("/photo-recommend", response_model=PhotoRecommendResponse)
async def photo_recommend(
    req: PhotoRecommendRequest,
    clip_tagger: ClipTagger = Depends(get_clip_tagger),
    song_curator: SongCurator = Depends(get_song_curator),
    reranker: Reranker = Depends(get_reranker),
    spotify_search: SpotifySearchClient = Depends(get_spotify_search_client),
) -> PhotoRecommendResponse:
    service = PhotoRecommendService(clip_tagger, song_curator, reranker, spotify_search)
    return await service.recommend(req)
