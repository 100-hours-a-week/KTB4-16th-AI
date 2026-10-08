"""② Moderation Service 진입점 — uvicorn app.moderation_main:app --port 8001"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api import moderation
from app.dependencies import get_moderation_model, verify_internal_token
from app.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 첫 채팅이 모델 로드(1~2초)를 기다리지 않게 서버 시작 때 미리 올린다
    get_moderation_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Muro AI Moderation", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(moderation.health_router)
    app.include_router(moderation.router, dependencies=[Depends(verify_internal_token)])
    return app


app = create_app()
