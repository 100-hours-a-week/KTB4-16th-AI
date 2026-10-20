from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifySearchClient
from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.reranker import Reranker
from app.components.saved_song_pool import SavedSongPool
from app.components.song_curator import SongCurator
from app.components.song_pool import SongPool
from app.config import get_settings
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.dependencies import (
    get_clip_tagger,
    get_embedder,
    get_reranker,
    get_saved_song_pool,
    get_song_curator,
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
    clip_tagger: ClipTagger = Depends(get_clip_tagger),
    song_curator: SongCurator = Depends(get_song_curator),
    reranker: Reranker = Depends(get_reranker),
    spotify_search: SpotifySearchClient = Depends(get_spotify_search_client),
    track_lookups: TrackLookupStore = Depends(get_track_lookup_store),
    song_pool: SongPool | None = Depends(get_song_pool),
    saved_pool: SavedSongPool = Depends(get_saved_song_pool),
    embedder: Embedder = Depends(get_embedder),
) -> PhotoRecommendResponse:
    s = get_settings()
    service = PhotoRecommendService(
        clip_tagger,
        song_curator,
        reranker,
        spotify_search,
        track_lookups,
        song_pool=song_pool,
        pool_count=s.song_pool_count,
        llm_count=s.song_pool_llm_count,
        saved_pool=saved_pool,
        saved_count=s.song_pool_saved_count,
        embedder=embedder,
    )
    return await service.recommend(req)
