from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.reranker import Reranker
from app.components.song_curator import SongCurator
from app.components.song_pool import SongPool
from app.components.vector_search import RecordSearch
from app.config import get_settings
from app.db.repositories.track_lookup_repository import TrackLookupStore
from app.dependencies import (
    get_embedder,
    get_record_search,
    get_reranker,
    get_song_curator,
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
    song_curator: SongCurator = Depends(get_song_curator),
    reranker: Reranker = Depends(get_reranker),
    spotify_search: SpotifySearchClient = Depends(get_spotify_search_client),
    track_lookups: TrackLookupStore = Depends(get_track_lookup_store),
    song_pool: SongPool | None = Depends(get_song_pool),
) -> ContextRecommendResponse:
    s = get_settings()
    service = ContextRecommendService(
        embedder,
        record_search,
        song_curator,
        reranker,
        spotify_search,
        track_lookups,
        song_pool=song_pool,
        pool_count=s.song_pool_count,
        llm_count=s.song_pool_llm_count,
        personal_similarity=s.personal_similarity,
    )
    return await service.recommend(req)
