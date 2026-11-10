"""LangSmith 추적 설정 — .env의 LANGSMITH_*·LANGCHAIN_* 값을 환경변수로 내보낸다.

LangSmith 라이브러리는 환경변수(os.environ)만 읽는다. 앱의 설정(app.config)은 .env를 자기 필드로만
읽고 환경변수로 내보내지 않아서, LANGSMITH_TRACING 같은 값을 .env에만 적어 두면 추적이 켜지지
않는다. 그래서 서버가 시작할 때(create_app 맨 앞) 이 두 이름으로 시작하는 값만 .env에서
환경변수로 옮긴다.

- 다른 이름(키·토큰 등)은 옮기지 않는다. LangSmith가 필요한 것만 환경변수가 된다.
- 이미 환경변수에 있는 값은 덮어쓰지 않는다. 서버·compose가 준 값이나 쉘에서 내보낸 값이 우선이다.
- LangSmith는 환경변수를 처음 읽을 때 값을 기억한다. 첫 추적 전에 옮겨야 해서 서버 시작 때 한다.
- 로그에는 옮긴 이름만 남기고 값은 남기지 않는다.
"""

import logging
import os
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger("muro.observability")

ENV_FILE = Path(".env")  # app.config의 Settings가 읽는 파일과 같다
PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
_TRACING_SWITCHES = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)
_API_KEYS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_HIDE_INPUTS = ("LANGSMITH_HIDE_INPUTS", "LANGCHAIN_HIDE_INPUTS")


def export_langsmith_env(env_file: Path | str = ENV_FILE) -> list[str]:
    """.env의 LangSmith 값을 환경변수로 옮기고, 옮긴 이름 목록을 돌려준다 (값은 돌려주지 않는다)."""
    path = Path(env_file)
    if not path.is_file():
        return []
    moved = []
    for name, value in dotenv_values(path).items():
        if not name.startswith(PREFIXES) or value is None or name in os.environ:
            continue
        os.environ[name] = value
        moved.append(name)
    if moved:
        logger.info("LangSmith 설정을 .env에서 환경변수로 옮겼어요: %s", ", ".join(moved))
    return moved


def _is_true(names: tuple[str, ...]) -> bool:
    return any(os.environ.get(name, "").strip().lower() == "true" for name in names)


def check_langsmith_setup() -> None:
    """추적이 켜져 있는데 안 되거나 위험한 설정이면 서버 로그에 경고한다. 꺼져 있으면 조용하다."""
    if not _is_true(_TRACING_SWITCHES):
        return
    if not any(os.environ.get(name) for name in _API_KEYS):
        logger.warning(
            "LangSmith 추적이 켜져 있지만 LANGSMITH_API_KEY가 없어서 추적이 올라가지 않아요"
        )
    if not _is_true(_HIDE_INPUTS):
        logger.warning(
            "LangSmith 추적이 켜져 있는데 LANGSMITH_HIDE_INPUTS가 true가 아니라서 "
            "질문 원문과 코멘트가 LangSmith로 올라가요"
        )


def setup_langsmith(env_file: Path | str = ENV_FILE) -> None:
    export_langsmith_env(env_file)
    check_langsmith_setup()
