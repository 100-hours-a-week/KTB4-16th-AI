from fastapi import APIRouter, Depends, Path, Response, status

from app.config import get_settings
from app.db.repositories.embedding_repository import EmbeddingStore
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.record_fact_repository import RecordFactRepository
from app.dependencies import get_embedding_store, get_job_repository, get_record_fact_repository
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


@router.delete("/embeddings/{recordId}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_embeddings(
    record_id: int = Path(alias="recordId", gt=0),
    store: EmbeddingStore = Depends(get_embedding_store),
    jobs: JobRepository = Depends(get_job_repository),
    facts: RecordFactRepository = Depends(get_record_fact_repository),
) -> Response:
    """자물쇠 삭제 시 백엔드가 호출. 대기 중인 작업과 저장된 벡터, 조건 검색용 값을 지운다.

    없는 자물쇠여도 204. 조건 검색용 값(record_facts)은 동기화도 지우지만, 이 호출로 바로 지워서
    삭제된 자물쇠가 검색에 나오는 시간을 없앤다.
    """
    await jobs.cancel(job_type=JOB_TYPE, dedupe_key=str(record_id))
    await store.delete(record_id)
    await facts.delete_many([record_id])
    return Response(status_code=status.HTTP_204_NO_CONTENT)
