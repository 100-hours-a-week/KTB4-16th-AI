"""③ Embedding Service 진입점 — uvicorn app.embedding_main:app --port 8002 (컨테이너 안에서만 열림)

텍스트 임베딩 모델(KURE-v1)을 한 번만 올려 두고 gateway·worker가 HTTP로 부르게 한다.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api import text_embed
from app.dependencies import get_kure_model, verify_internal_token
from app.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 요청이 모델 로딩(수 초~수십 초)을 기다리지 않게 서버 시작 때 미리 올린다.
    # 이 프로세스가 로딩하는 동안에도 gateway(8000)의 /health는 영향을 받지 않는다.
    get_kure_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Muro AI Embedding", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(text_embed.health_router)
    app.include_router(text_embed.router, dependencies=[Depends(verify_internal_token)])
    return app


app = create_app()
