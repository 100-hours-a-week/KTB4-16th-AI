from fastapi import APIRouter, Depends

from app.clients.image_fetcher import ImageFetcher
from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.scene_curator import SceneCurator
from app.components.song_pool import SongPool
from app.config import get_settings
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.dependencies import (
    get_embedder,
    get_image_fetcher,
    get_scene_curator,
    get_song_pool,
    get_spotify_search_client,
    get_track_lookup_store,
)
from app.schemas.photo_recommend import PhotoRecommendRequest, PhotoRecommendResponse
from app.services.photo_recommend_service import PhotoRecommendService

router = APIRouter(prefix="/api", tags=["기능3 사진 기반 추천"])


@router.post("/photo-recommend", response_model=PhotoRecommendResponse)
async def photo_recommend(
    req: PhotoRecommendRequest,
    scene_curator: SceneCurator = Depends(get_scene_curator),
    spotify_search: SpotifySearchClient = Depends(get_spotify_search_client),
    track_lookups: TrackLookupStore = Depends(get_track_lookup_store),
    song_pool: SongPool | None = Depends(get_song_pool),
    embedder: Embedder = Depends(get_embedder),
    image_fetcher: ImageFetcher = Depends(get_image_fetcher),
) -> PhotoRecommendResponse:
    service = PhotoRecommendService(
        scene_curator,
        spotify_search,
        embedder,
        image_fetcher,
        track_lookups,
        song_pool=song_pool,
        hub_weight=get_settings().hub_weight,
    )
    return await service.recommend(req)
