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

    async def mark_done_by_dedupe_key(self, *, job_type: str, dedupe_key: str) -> None:
        """(job_type, dedupe_key)로 직접 done 처리. job_id를 모르는 호출부(핸들러 내부)가
        "배치의 남은 개수"를 정확히 세려면, worker_main의 mark_done을 기다리지 않고
        여기서 먼저 자기 상태를 반영해야 한다 (그래야 나중의 count_incomplete가 자신을
        빼고 셀 수 있음). worker_main이 이후 호출할 mark_done(job_id)은 이미 done이라
        조건에 안 걸려 조용히 아무 일도 안 한다."""
        await self._session.execute(
            text(
                "UPDATE ai_jobs SET status = 'done', last_error = NULL, updated_at = now() "
                "WHERE job_type = :job_type AND dedupe_key = :key AND status = 'running'"
            ),
            {"job_type": job_type, "key": dedupe_key},
        )
        await self._session.commit()

    async def count_incomplete(self, *, job_type: str, dedupe_key_prefix: str) -> int:
        """dedupe_key가 이 접두사로 시작하는 job_type 작업 중 아직 안 끝난(pending·running)
        개수. failed는 "재시도 다 써서 포기한 것"이라 미완료로 안 센다 — 그거 하나 때문에
        배치 완료 알림이 영원히 안 나가면 안 됨."""
        row = await self._session.execute(
            text(
                """
                SELECT COUNT(*) FROM ai_jobs
                WHERE job_type = :job_type AND dedupe_key LIKE :prefix
                  AND status IN ('pending', 'running')
                """
            ),
            {"job_type": job_type, "prefix": f"{dedupe_key_prefix}%"},
        )
        return row.scalar_one()

    async def completed_user_ids(self, *, job_type: str, dedupe_key_prefix: str) -> list[int]:
        """이 배치에서 성공(done)한 job들의 payload.user_id 목록."""
        rows = await self._session.execute(
            text(
                """
                SELECT payload ->> 'user_id' AS user_id FROM ai_jobs
                WHERE job_type = :job_type AND dedupe_key LIKE :prefix AND status = 'done'
                """
            ),
            {"job_type": job_type, "prefix": f"{dedupe_key_prefix}%"},
        )
        # payload ->> 'user_id'는 문자열로 나온다. 백엔드 userId는 숫자(Long)라 되돌린다.
        return [int(row.user_id) for row in rows if row.user_id is not None]

    async def try_claim_once(self, *, job_type: str, dedupe_key: str) -> bool:
        """(job_type, dedupe_key) 조합의 "최초 1회 실행권"을 얻는다.

        배치 완료 알림처럼 "동시에 여러 job이 끝나도 딱 한 번만 실행돼야 하는 일"의
        중복 실행을 막는 데 쓴다. ai_jobs의 기존 유니크 제약(uq_ai_jobs_type_key)을
        그대로 이용 — 이미 있으면 아무것도 안 하고 False, 처음이면 행을 만들고 True.
        """
        stmt = (
            insert(AiJob)
            .values(
                job_type=job_type, dedupe_key=dedupe_key, payload={}, status="done", max_attempts=1
            )
            .on_conflict_do_nothing(constraint="uq_ai_jobs_type_key")
        )
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount > 0
