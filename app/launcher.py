"""전체 실행 — `python -m app.main` 한 줄로 AI 서버 전체를 띄운다 (클라우드 배포용).

순서:
  1. DB 마이그레이션 (alembic upgrade head). DB가 아직 안 떴으면 잠깐 기다렸다 다시 시도
  2. ① gateway ② moderation ③ worker 를 동시에 띄움

셋 중 하나라도 죽으면 나머지도 내리고 컨테이너를 종료한다. 반쯤 살아있는 컨테이너로
남으면 밖에서는 정상처럼 보이는데 기능 일부만 안 되는 상태가 되기 때문이다 — 종료되면
도커 재시작 정책이 컨테이너째 다시 띄운다.
"""

import logging
import signal
import subprocess
import sys
import time

from app.config import get_settings

logger = logging.getLogger("muro.launcher")

MIGRATE_ATTEMPTS = 10
MIGRATE_RETRY_SECONDS = 3
POLL_SECONDS = 1
STOP_TIMEOUT_SECONDS = 10


def commands() -> dict[str, list[str]]:
    settings = get_settings()
    uvicorn = [sys.executable, "-m", "uvicorn", "--host", "0.0.0.0"]
    return {
        "gateway": [*uvicorn, "app.main:app", "--port", str(settings.gateway_port)],
        "moderation": [
            *uvicorn,
            "app.moderation_main:app",
            "--port",
            str(settings.moderation_port),
        ],
        "worker": [sys.executable, "-m", "app.worker_main"],
    }


def migrate() -> None:
    for attempt in range(1, MIGRATE_ATTEMPTS + 1):
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"])
        if result.returncode == 0:
            logger.info("마이그레이션 완료")
            return
        logger.warning(
            "마이그레이션 실패 (%s/%s) — %s초 뒤 재시도",
            attempt,
            MIGRATE_ATTEMPTS,
            MIGRATE_RETRY_SECONDS,
        )
        time.sleep(MIGRATE_RETRY_SECONDS)
    raise SystemExit("마이그레이션이 계속 실패해서 서버를 띄우지 않음 (DATABASE_URL 확인)")


def supervise(cmds: dict[str, list[str]]) -> int:
    """cmds를 모두 띄우고, 하나가 죽거나 종료 신호가 오면 전부 내린다. 종료 코드를 돌려준다."""
    stop_requested = False

    def _on_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        logger.info("종료 신호(%s) 받음 — 전체 종료", signal.Signals(signum).name)
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    procs = {name: subprocess.Popen(cmd) for name, cmd in cmds.items()}
    logger.info("실행: %s", ", ".join(procs))

    exit_code = 0
    while not stop_requested:
        dead = [(name, p.returncode) for name, p in procs.items() if p.poll() is not None]
        if dead:
            name, code = dead[0]
            logger.error("%s 가 종료됨 (코드 %s) — 나머지도 내린다", name, code)
            # 계속 떠 있어야 하는 프로세스라 코드 0으로 끝나도 비정상 종료로 본다.
            # 신호로 죽었으면(음수) 셸 관례대로 128+신호번호 (kill -9 → 137)
            exit_code = 128 - code if code < 0 else code or 1
            break
        time.sleep(POLL_SECONDS)

    _stop_all(procs)
    return exit_code


def _stop_all(procs: dict[str, subprocess.Popen]) -> None:
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    for name, p in procs.items():
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            logger.warning("%s 가 %s초 안에 안 꺼져서 강제 종료", name, STOP_TIMEOUT_SECONDS)
            p.kill()
            p.wait()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    migrate()
    sys.exit(supervise(commands()))
