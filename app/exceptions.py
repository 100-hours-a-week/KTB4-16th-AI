"""공통 예외와 에러 응답 포맷 `{code, message, field}`."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


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


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.field),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(part) for part in first.get("loc", ()) if part != "body"]
        field = ".".join(loc) or None
        return JSONResponse(
            status_code=InvalidInputError.status_code,
            content=_error_body(InvalidInputError.code, InvalidInputError.message, field),
        )
