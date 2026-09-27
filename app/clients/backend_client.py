"""AI → 백엔드 콜백 어댑터.

지금까지 모든 통신은 백엔드 → AI 방향이었다. RECAP 배치 생성 완료를 AI가
먼저 백엔드에 알리는 게 유일하게 반대 방향인 호출이다.

경로·바디는 AI팀 자체 API 시트("AI -> BE" 탭) 기준 — 백엔드 최종 확정 전.
이 콜백은 "다 됐다"는 알림일 뿐, 요약·통계 내용 자체는 안 싣는다(백엔드가
알림함 항목만 생성하면 되므로). 실제 내용을 백엔드가 어떻게 가져가는지는
별도 조회 API가 필요해 보이는데, 아직 설계·확정 전이라 여기 없다.
"""

from typing import Protocol

import httpx

from app.exceptions import UpstreamError


class BackendCallbackError(UpstreamError):
    message = "백엔드 콜백 호출에 실패했어요."


class ReportBackendClient(Protocol):
    async def notify_report_ready(
        self, *, year: int, month: int, user_ids: list[int], generated_at: str
    ) -> None: ...


class BackendClient:
    def __init__(self, base_url: str, internal_token: str, timeout: float):
        self._internal_token = internal_token
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def notify_report_ready(
        self, *, year: int, month: int, user_ids: list[int], generated_at: str
    ) -> None:
        """RECAP 배치 생성 완료를 백엔드에 알린다 (AI API 시트 "AI -> BE" 탭 기준)."""
        try:
            res = await self._http.post(
                "/internal/ai/report-ready",
                json={
                    "year": year,
                    "month": month,
                    "userIds": user_ids,
                    "generatedAt": generated_at,
                },
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
        self, *, year: int, month: int, user_ids: list[int], generated_at: str
    ) -> None:
        self.calls.append(
            {"year": year, "month": month, "user_ids": user_ids, "generated_at": generated_at}
        )
