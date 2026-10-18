import sys

import pytest

from app.config import get_settings
from app.launcher import commands, supervise


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_one_process_dies_stops_the_rest_and_returns_its_code():
    """하나라도 죽으면 반쯤 살아있는 컨테이너로 남지 않게 전체를 내린다."""
    code = supervise(
        {
            "long": _py("import time; time.sleep(60)"),
            "crash": _py("import sys, time; time.sleep(0.2); sys.exit(3)"),
        }
    )

    assert code == 3


def test_process_exiting_normally_is_still_treated_as_failure():
    """계속 떠 있어야 하는 프로세스가 코드 0으로 끝나도 비정상으로 본다."""
    code = supervise({"long": _py("import time; time.sleep(60)"), "quit": _py("pass")})

    assert code == 1


@pytest.fixture
def run_worker_env(monkeypatch):
    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv("RUN_WORKER", raising=False)
        else:
            monkeypatch.setenv("RUN_WORKER", value)
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


def test_worker_runs_by_default(run_worker_env):
    run_worker_env(None)
    assert set(commands()) == {"gateway", "moderation", "embedding", "worker"}


def test_run_worker_false_starts_only_api_servers(run_worker_env):
    """블루그린 대기 VM — 워커가 공유 DB 큐에서 운영 작업을 가져가지 않게 뺀다."""
    run_worker_env("false")
    assert set(commands()) == {"gateway", "moderation", "embedding"}


def test_embedding_service_listens_only_inside_the_container(run_worker_env):
    """임베딩 서비스는 gateway·worker만 부르므로 바깥에 열지 않는다."""
    run_worker_env(None)
    cmd = commands()["embedding"]

    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8002"


def test_no_embedding_process_when_provider_is_openai(monkeypatch):
    monkeypatch.setenv("TEXT_EMBEDDING_PROVIDER", "openai")
    get_settings.cache_clear()
    try:
        assert "embedding" not in commands()
    finally:
        get_settings.cache_clear()
