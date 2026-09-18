from fastapi import APIRouter, Depends, status

from app.config import get_settings
from app.db.repositories.job_repository import JobRepository
from app.dependencies import get_job_repository
from app.schemas.common import QueuedResponse
from app.schemas.embeddings import EmbeddingGenerateRequest

router = APIRouter(prefix="/api", tags=["기능4 임베딩 인덱싱"])

JOB_TYPE = "embedding"


@router.post(
    "/embeddings/generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=QueuedResponse,
)
async def generate_embeddings(
    req: EmbeddingGenerateRequest,
    jobs: JobRepository = Depends(get_job_repository),
) -> QueuedResponse:
    """큐에 등록만 하고 즉시 202. 실제 임베딩은 워커가 처리한다."""
    job_id = await jobs.enqueue(
        job_type=JOB_TYPE,
        dedupe_key=str(req.record_id),
        payload=req.model_dump(mode="json", by_alias=True),
        max_attempts=get_settings().job_max_attempts,
    )
    return QueuedResponse(job_id=f"emb_{job_id}")
