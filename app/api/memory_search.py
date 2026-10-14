from fastapi import APIRouter, Depends

from app.dependencies import get_memory_search_service
from app.schemas.memory_search import MemorySearchRequest, MemorySearchResponse
from app.services.memory_search_service import MemorySearchService

router = APIRouter(prefix="/api", tags=["기능5 기억 검색"])


@router.post("/memory-search", response_model=MemorySearchResponse)
async def memory_search(
    req: MemorySearchRequest,
    service: MemorySearchService = Depends(get_memory_search_service),
) -> MemorySearchResponse:
    return await service.search(req)
