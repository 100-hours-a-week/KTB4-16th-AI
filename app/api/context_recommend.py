from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.scene_curator import SceneCurator
from app.components.song_pool import SongPool
from app.components.taste_genres import OnboardingReader
from app.components.vector_search import RecordSearch
from app.config import get_settings
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.dependencies import (
    get_embedder,
    get_onboarding_reader,
    get_record_search,
    get_scene_curator,
    get_song_pool,
    get_spotify_search_client,
    get_track_lookup_store,
)
from app.schemas.context_recommend import ContextRecommendRequest, ContextRecommendResponse
from app.services.context_recommend_service import ContextRecommendService

router = APIRouter(prefix="/api", tags=["기능1 상황 맞춤 추천"])


@router.post("/context-recommend", response_model=ContextRecommendResponse)
async def context_recommend(
    req: ContextRecommendRequest,
    embedder: Embedder = Depends(get_embedder),
    record_search: RecordSearch = Depends(get_record_search),
    scene_curator: SceneCurator = Depends(get_scene_curator),
    spotify_search: SpotifySearchClient = Depends(get_spotify_search_client),
    track_lookups: TrackLookupStore = Depends(get_track_lookup_store),
    song_pool: SongPool | None = Depends(get_song_pool),
    onboarding: OnboardingReader = Depends(get_onboarding_reader),
) -> ContextRecommendResponse:
    s = get_settings()
    service = ContextRecommendService(
        embedder,
        record_search,
        scene_curator,
        spotify_search,
        track_lookups,
        song_pool=song_pool,
        personal_similarity=s.personal_similarity,
        nearby_weight=s.nearby_weight,
        hub_weight=s.hub_weight,
        onboarding=onboarding,
    )
    return await service.recommend(req)
