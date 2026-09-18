"""② Moderation Service 진입점 — uvicorn app.moderation_main:app --port 8001"""

from fastapi import Depends, FastAPI

from app.api import moderation
from app.dependencies import verify_internal_token
from app.exceptions import register_exception_handlers


def create_app() -> FastAPI:
    app = FastAPI(title="Muro AI Moderation", version="0.1.0")
    register_exception_handlers(app)
    app.include_router(moderation.health_router)
    app.include_router(moderation.router, dependencies=[Depends(verify_internal_token)])
    return app


app = create_app()
