"""공통 예외와 에러 응답 포맷 `{code, message, field}`."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("muro.api")


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"
    message = "서버 내부 오류가 발생했어요."

    def __init__(self, message: str | None = None, field: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message
        self.field = field


class InvalidInputError(AppError):
    status_code = 400
    code = "INVALID_INPUT"
    message = "요청 값이 올바르지 않아요."


class UnauthorizedError(AppError):
    status_code = 401
    code = "UNAUTHORIZED"
    message = "내부 인증 토큰이 없거나 올바르지 않아요."


class NotImplementedFeatureError(AppError):
    status_code = 501
    code = "NOT_IMPLEMENTED"
    message = "아직 구현되지 않은 기능이에요."


class UpstreamError(AppError):
    status_code = 503
    code = "UPSTREAM_UNAVAILABLE"
    message = "외부 서비스와 통신이 원활하지 않아요."


class ClipError(UpstreamError):
    message = "CLIP API 호출에 실패했어요."


class LLMError(UpstreamError):
    message = "LLM API 호출에 실패했어요."


class EmbeddingError(UpstreamError):
    message = "임베딩 API 호출에 실패했어요."


class ModelUnavailableError(AppError):
    status_code = 503
    code = "MODEL_UNAVAILABLE"
    message = "모델을 사용할 수 없어요."


def _error_body(code: str, message: str, field: str | None) -> dict:
    return {"code": code, "message": message, "field": field}


def _describe(error: dict) -> str:
    loc = ".".join(str(part) for part in error.get("loc", ()))
    return f"{loc or '(바디 전체)'} — {error.get('msg', '')}"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.field),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # 400만 보이면 백엔드도 클라우드도 원인을 못 찾는다(실제로 겪음). 어느 필드가 왜
        # 틀렸는지와 Content-Type을 남긴다 — 보낸 값(서명 URL·코멘트 등)은 남기지 않는다.
        logger.warning(
            "요청 검증 실패 %s %s (Content-Type: %s): %s",
            request.method,
            request.url.path,
            request.headers.get("content-type") or "없음",
            "; ".join(_describe(e) for e in exc.errors()) or "상세 없음",
        )
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(part) for part in first.get("loc", ()) if part not in ("body", "path", "query")]
        field = ".".join(loc) or None
        return JSONResponse(
            status_code=InvalidInputError.status_code,
            content=_error_body(InvalidInputError.code, InvalidInputError.message, field),
        )
