"""ai_jobs 기반 작업 큐 — 게이트웨이가 넣고 워커가 꺼낸다.

별도 브로커 없이 AI PostgreSQL 하나로 프로세스 간 전달·재시도·상태 조회를 처리한다.
나중에 Pub/Sub 등으로 옮기면 이 파일과 worker_main만 바뀐다.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AiJob

RETRY_BASE_SECONDS = 30


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    job_type: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


class JobRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def enqueue(
        self, *, job_type: str, dedupe_key: str, payload: dict[str, Any], max_attempts: int
    ) -> int:
        """같은 (job_type, dedupe_key)가 다시 오면 새 payload로 대기 상태를 초기화한다."""
        stmt = insert(AiJob).values(
            job_type=job_type,
            dedupe_key=dedupe_key,
            payload=payload,
            max_attempts=max_attempts,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ai_jobs_type_key",
            set_={
                "payload": stmt.excluded.payload,
                "max_attempts": stmt.excluded.max_attempts,
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "run_after": func.now(),
                "updated_at": func.now(),
            },
        ).returning(AiJob.id)
        job_id = (await self._session.execute(stmt)).scalar_one()
        await self._session.commit()
        return job_id

    async def claim(self, *, stale_after_seconds: int) -> ClaimedJob | None:
        """실행할 작업 1건을 잠그고 running으로 바꾼다.

        워커가 죽어 running으로 멈춘 작업은 stale_after_seconds가 지나면 다시 가져간다.
        """
        row = (
            await self._session.execute(
                text(
                    """
                    UPDATE ai_jobs
                    SET status = 'running', attempts = attempts + 1, updated_at = now()
                    WHERE id = (
                        SELECT id FROM ai_jobs
                        WHERE (status = 'pending' AND run_after <= now())
                           OR (status = 'running'
                               AND updated_at < now() - make_interval(secs => :stale))
                        ORDER BY run_after, id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING id, job_type, payload, attempts, max_attempts
                    """
                ),
                {"stale": stale_after_seconds},
            )
        ).first()
        await self._session.commit()
        if row is None:
            return None
        return ClaimedJob(
            id=row.id,
            job_type=row.job_type,
            payload=row.payload,
            attempts=row.attempts,
            max_attempts=row.max_attempts,
        )

    async def mark_done(self, job_id: int) -> None:
        await self._session.execute(
            text(
                "UPDATE ai_jobs SET status = 'done', last_error = NULL, updated_at = now() "
                "WHERE id = :id AND status = 'running'"
            ),
            {"id": job_id},
        )
        await self._session.commit()

    async def mark_failed(self, job: ClaimedJob, error: str) -> None:
        """재시도가 남으면 지수 백오프 후 pending, 아니면 failed."""
        retry = job.attempts < job.max_attempts
        delay = RETRY_BASE_SECONDS * 2 ** (job.attempts - 1)
        await self._session.execute(
            text(
                """
                UPDATE ai_jobs
                SET status = :status,
                    last_error = :error,
                    run_after = now() + make_interval(secs => :delay),
                    updated_at = now()
                WHERE id = :id AND status = 'running'
                """
            ),
            {
                "id": job.id,
                "status": "pending" if retry else "failed",
                "error": error[:2000],
                "delay": delay if retry else 0,
            },
        )
        await self._session.commit()

    async def cancel(self, *, job_type: str, dedupe_key: str) -> None:
        """작업을 cancelled로 표시한다. 워커는 cancelled 작업을 꺼내지 않고,
        이미 실행 중이던 작업은 저장 직전에 이 표시를 보고 저장을 건너뛴다."""
        await self._session.execute(
            text(
                "UPDATE ai_jobs SET status = 'cancelled', updated_at = now() "
                "WHERE job_type = :job_type AND dedupe_key = :key"
            ),
            {"job_type": job_type, "key": dedupe_key},
        )
        await self._session.commit()
