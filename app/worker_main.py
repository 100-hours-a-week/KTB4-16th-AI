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
from app.workers.types import JobHandler

logger = logging.getLogger("muro.worker")

HANDLERS: dict[str, JobHandler] = {
    embedding_worker.JOB_TYPE: embedding_worker.handle,
    report_worker.JOB_TYPE: report_worker.handle,
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
            await JobRepository(session).mark_failed(job, f"{type(e).__name__}: {e}")
        return

    async with sessionmaker() as session:
        await JobRepository(session).mark_done(job.id)
    logger.info("job %s 완료 (%s)", job.id, job.job_type)


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
