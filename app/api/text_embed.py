import time

from fastapi import APIRouter, Depends

from app.components.kure_model import TextEmbeddingModel
from app.dependencies import get_kure_model
from app.exceptions import ModelUnavailableError
from app.schemas.health import EmbeddingHealthResponse
from app.schemas.text_embed import EmbedRequest, EmbedResponse

# 인증이 필요한 라우터와 헬스체크 라우터를 분리한다 (모더레이션과 같은 구조)
router = APIRouter(prefix="/api", tags=["텍스트 임베딩 (내부)"])
health_router = APIRouter(prefix="/api/embed", tags=["health"])

_started_at = time.monotonic()


# 모델 추론은 CPU를 오래 쓰는 동기 코드라서 async가 아니라 일반 함수로 둔다.
# FastAPI가 스레드풀에서 실행해서 다른 요청(헬스체크 등)이 이벤트 루프에서 막히지 않는다.
@router.post("/embed", response_model=EmbedResponse)
def embed(
    req: EmbedRequest, model: TextEmbeddingModel | None = Depends(get_kure_model)
) -> EmbedResponse:
    if model is None:
        raise ModelUnavailableError("임베딩 모델 파일이 없어요.")
    return EmbedResponse(model=model.model, dim=model.dim, vectors=model.embed(req.texts))


@health_router.get("/health", response_model=EmbeddingHealthResponse)
def embedding_health(
    model: TextEmbeddingModel | None = Depends(get_kure_model),
) -> EmbeddingHealthResponse:
    return EmbeddingHealthResponse(
        status="ok",
        model=model.model if model else None,
        dim=model.dim if model else None,
        uptime_seconds=int(time.monotonic() - _started_at),
    )
