from fastapi import APIRouter, Depends

from app.clients.spotify_client import SpotifySearchClient
from app.components.embedder import Embedder
from app.components.reranker import Reranker
from app.components.song_curator import SongCurator
from app.components.vector_search import RecordSearch
from app.dependencies import (
    get_embedder,
    get_record_search,
    get_reranker,
    get_song_curator,
    get_spotify_search_client,
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
) -> ContextRecommendResponse:
    service = ContextRecommendService(
        embedder, record_search, song_curator, reranker, spotify_search
    )
    return await service.recommend(req)
