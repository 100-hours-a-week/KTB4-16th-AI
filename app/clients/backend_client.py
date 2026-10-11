"""AI → 백엔드 콜백 어댑터.

지금까지 모든 통신은 백엔드 → AI 방향이었다. RECAP 배치 생성 완료를 AI가
먼저 백엔드에 알리는 게 유일하게 반대 방향인 호출이다.

바디는 백엔드 DTO(MonthlyReportAiCallbackRequest) 확정 형식 — 콜백에 완료
알림뿐 아니라 각 유저의 실제 리포트 내용(aiRecap, photoScenes)까지 실어
보낸다. 별도 조회 API 없이 이 콜백 하나로 백엔드가 필요한 내용을 전부 받는다.
"""

from typing import Any, Protocol

import httpx

from app.exceptions import UpstreamError


class BackendCallbackError(UpstreamError):
    message = "백엔드 콜백 호출에 실패했어요."


class ReportBackendClient(Protocol):
    async def notify_report_ready(
        self,
        *,
        job_id: str,
        year: int,
        month: int,
        generated_at: str,
        results: list[dict[str, Any]],
        batch_request_id: str | None = None,
    ) -> None: ...


class BackendClient:
    def __init__(self, base_url: str, internal_token: str, timeout: float):
        self._internal_token = internal_token
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def notify_report_ready(
        self,
        *,
        job_id: str,
        year: int,
        month: int,
        generated_at: str,
        results: list[dict[str, Any]],
        batch_request_id: str | None = None,
    ) -> None:
        """RECAP 배치 생성 완료 + 결과 내용을 백엔드에 알린다 (MonthlyReportAiCallbackRequest)."""
        body: dict[str, Any] = {
            "jobId": job_id,
            "year": year,
            "month": month,
            "generatedAt": generated_at,
            "results": results,
        }
        # 요청 때 받은 batchRequestId를 값 그대로 돌려준다. 없으면(구버전 백엔드 요청) 필드를
        # 아예 빼서 기존 본문과 똑같이 보낸다 — null을 보내면 백엔드가 "모르는 ID"로 볼 수 있다.
        if batch_request_id is not None:
            body["batchRequestId"] = batch_request_id
        try:
            res = await self._http.post(
                "/internal/ai/report-ready",
                json=body,
                headers={"X-Internal-Token": self._internal_token},
            )
        except httpx.HTTPError as e:
            raise BackendCallbackError(f"백엔드 콜백 연결 실패: {type(e).__name__}") from e
        if res.status_code >= 400:
            raise BackendCallbackError(
                f"백엔드 콜백 실패 (status {res.status_code}): {res.text[:200]}"
            )


class FakeBackendClient:
    """fake 모드·테스트용 — 실제 HTTP 호출 없이 호출 내역만 기록한다."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def notify_report_ready(
        self,
        *,
        job_id: str,
        year: int,
        month: int,
        generated_at: str,
        results: list[dict[str, Any]],
        batch_request_id: str | None = None,
    ) -> None:
        self.calls.append(
            {
                "job_id": job_id,
                "year": year,
                "month": month,
                "generated_at": generated_at,
                "results": results,
                "batch_request_id": batch_request_id,
            }
        )
