"""기능2 RECAP 배치 처리 순서."""

from app.config import get_settings
from app.db.mysql import get_sessionmaker as get_mysql_sessionmaker
from app.db.postgres import get_sessionmaker as get_postgres_sessionmaker
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.report_repository import ReportRepository
from app.schemas.reports import ReportBatchGenerateRequest, ReportBatchQueuedResponse

JOB_TYPE = "report"


class ReportService:
    def __init__(self, jobs: JobRepository):
        self._jobs = jobs

    async def enqueue_batch(self, req: ReportBatchGenerateRequest) -> ReportBatchQueuedResponse:
        user_ids = await self._resolve_target_user_ids(req)

        max_attempts = get_settings().job_max_attempts
        for user_id in user_ids:
            await self._jobs.enqueue(
                job_type=JOB_TYPE,
                dedupe_key=f"{req.year}-{req.month}-{user_id}",
                payload={"user_id": user_id, "year": req.year, "month": req.month},
                max_attempts=max_attempts,
            )

        # 배치 하나에 job이 여러 개 걸리므로, 개별 job_id 대신 배치를 가리키는 합성 id를 준다.
        return ReportBatchQueuedResponse(
            job_id=f"report_batch_{req.year}-{req.month}",
            target_count=len(user_ids),
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
