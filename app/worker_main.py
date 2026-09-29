"""③ Background Worker 진입점 — python -m app.worker_main

ai_jobs 큐를 폴링해 job_type별 핸들러로 분기한다. 포트·FastAPI 없음.
"""

import asyncio
import logging
import signal

from app.config import get_settings
from app.db.postgres import get_engine, get_sessionmaker
from app.db.repositories.job_repository import ClaimedJob, JobRepository
from app.workers import embedding_worker, report_worker
from app.workers.types import JobFinishedHook, JobHandler

logger = logging.getLogger("muro.worker")

HANDLERS: dict[str, JobHandler] = {
    embedding_worker.JOB_TYPE: embedding_worker.handle,
    report_worker.JOB_TYPE: report_worker.handle,
}

# 성공이든 최종 실패든 job이 끝나면 부른다. 핸들러 안(성공 경로)에만 두면 실패로
# 끝난 job은 후처리를 못 탄다 — 배치의 마지막 job이 실패하면 완료 알림이 영영 안
# 나가던 문제(2026-09 RECAP)가 이것 때문이었다.
ON_FINISHED: dict[str, JobFinishedHook] = {
    report_worker.JOB_TYPE: report_worker.on_finished,
}


async def run_once() -> bool:
    """작업 1건 처리. 처리할 작업이 없으면 False."""
    settings = get_settings()
    sessionmaker = get_sessionmaker()

    async with sessionmaker() as session:
        job = await JobRepository(session).claim(
            stale_after_seconds=settings.job_stale_after_seconds
        )
    if job is None:
        return False

    await _execute(job)
    return True


async def _execute(job: ClaimedJob) -> None:
    sessionmaker = get_sessionmaker()
    handler = HANDLERS.get(job.job_type)
    try:
        if handler is None:
            raise RuntimeError(f"알 수 없는 job_type: {job.job_type}")
        await handler(job.payload)
    except Exception as e:
        logger.exception("job %s 실패 (시도 %s/%s)", job.id, job.attempts, job.max_attempts)
        async with sessionmaker() as session:
            gave_up = await JobRepository(session).mark_failed(job, f"{type(e).__name__}: {e}")
        if gave_up:
            await _on_finished(job)
        return

    async with sessionmaker() as session:
        await JobRepository(session).mark_done(job.id)
    logger.info("job %s 완료 (%s)", job.id, job.job_type)
    await _on_finished(job)


async def _on_finished(job: ClaimedJob) -> None:
    """상태(done·failed)를 먼저 확정한 뒤에 부른다 — 그래야 후처리가 "남은 job"을
    셀 때 이 job을 끝난 것으로 본다. 후처리가 실패해도 job 자체의 결과는 그대로 둔다."""
    hook = ON_FINISHED.get(job.job_type)
    if hook is None:
        return
    try:
        await hook(job.payload)
    except Exception:
        logger.exception("job %s 후처리 실패 (%s)", job.id, job.job_type)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    logger.info("worker 시작 (client mode: %s)", settings.ai_client_mode)
    while not stop.is_set():
        try:
            processed = await run_once()
        except Exception:
            logger.exception("큐 조회 실패")
            processed = False
        if not processed:
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_interval_seconds)
            except TimeoutError:
                pass

    await get_engine().dispose()
    logger.info("worker 종료")


if __name__ == "__main__":
    asyncio.run(main())
