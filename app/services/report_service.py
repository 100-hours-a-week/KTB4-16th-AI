"""기능2 RECAP 배치 처리 순서."""

from app.config import get_settings
from app.db.mysql import get_sessionmaker as get_mysql_sessionmaker
from app.db.postgres import get_sessionmaker as get_postgres_sessionmaker
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.report_repository import ReportRepository
from app.schemas.reports import ReportBatchGenerateRequest, ReportBatchQueuedResponse
from app.workers.report_worker import BATCH_NOTIFY_JOB_TYPE, BatchScope

JOB_TYPE = "report"


class ReportService:
    def __init__(self, jobs: JobRepository):
        self._jobs = jobs

    async def enqueue_batch(self, req: ReportBatchGenerateRequest) -> ReportBatchQueuedResponse:
        scope = BatchScope(year=req.year, month=req.month, batch_request_id=req.batch_request_id)

        if req.batch_request_id:
            # 같은 batchRequestId로 요청이 또 왔다(백엔드 재시도·중복 전송). 이미 접수한 배치라
            # 다시 큐에 넣지 않고 "접수됨"만 돌려준다 — 새 알림이 또 나가지 않게.
            already = await self._jobs.count_with_prefix(
                job_type=JOB_TYPE, dedupe_key_prefix=scope.job_prefix
            )
            if already > 0:
                return self._queued(req, target_count=already)
        else:
            # batchRequestId가 없는 요청(구버전 백엔드)은 연·월로 구분한다. 같은 달을 다시
            # 돌리면(QA 재생성 등) 지난 배치의 "알림 1회 전송권"이 남아 있어 이번 배치가
            # 끝나도 콜백이 안 나간다 — 새 배치를 넣기 전에 돌려놓는다.
            # ID가 있는 요청은 배치마다 알림 키가 달라서 돌려놓을 게 없다.
            await self._jobs.release_once(
                job_type=BATCH_NOTIFY_JOB_TYPE, dedupe_key=scope.notify_key
            )

        user_ids = await self._resolve_target_user_ids(req)

        max_attempts = get_settings().job_max_attempts
        for user_id in user_ids:
            await self._jobs.enqueue(
                job_type=JOB_TYPE,
                dedupe_key=scope.job_key(user_id),
                payload={
                    "user_id": user_id,
                    "year": req.year,
                    "month": req.month,
                    "batch_request_id": req.batch_request_id,
                },
                max_attempts=max_attempts,
            )

        return self._queued(req, target_count=len(user_ids))

    @staticmethod
    def _queued(req: ReportBatchGenerateRequest, *, target_count: int) -> ReportBatchQueuedResponse:
        # 배치 하나에 job이 여러 개 걸리므로, 개별 job_id 대신 배치를 가리키는 합성 id를 준다.
        # 형태는 예전 그대로(백엔드가 이미 쓰고 있음) — 배치 구분은 batchRequestId가 맡는다.
        return ReportBatchQueuedResponse(
            job_id=f"report_batch_{req.year}-{req.month}",
            target_count=target_count,
            batch_request_id=req.batch_request_id,
        )

    async def _resolve_target_user_ids(self, req: ReportBatchGenerateRequest) -> list[int]:
        if req.user_ids is not None:
            return req.user_ids

        async with (
            get_mysql_sessionmaker()() as mysql_session,
            get_postgres_sessionmaker()() as pg_session,
        ):
            repo = ReportRepository(mysql_session, pg_session)
            return await repo.get_active_user_ids(year=req.year, month=req.month)
