"""① AI Gateway 진입점 — uvicorn app.main:app --port 8000"""

from fastapi import Depends, FastAPI

from app.api import (
    balance,
    context_recommend,
    embeddings,
    health,
    memory_search,
    photo_recommend,
    playlist,
    reports,
)
from app.dependencies import verify_internal_token
from app.exceptions import register_exception_handlers


def create_app() -> FastAPI:
    app = FastAPI(title="Muro AI Gateway", version="0.1.0")
    register_exception_handlers(app)

    app.include_router(health.router)
    for module in (
        embeddings,
        photo_recommend,
        context_recommend,
        playlist,
        reports,
        memory_search,
        balance,
    ):
        app.include_router(module.router, dependencies=[Depends(verify_internal_token)])
    return app


app = create_app()
