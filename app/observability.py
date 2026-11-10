"""LangSmith 추적 설정 — .env의 LANGSMITH_*·LANGCHAIN_* 값을 환경변수로 내보낸다.


LangSmith 라이브러리는 환경변수(os.environ)만 읽는다. 앱의 설정(app.config)은 .env를 자기 필드로만
읽고 환경변수로 내보내지 않아서, LANGSMITH_TRACING 같은 값을 .env에만 적어 두면 추적이 켜지지
않는다. 그래서 서버가 시작할 때(create_app 맨 앞) 이 두 이름으로 시작하는 값만 .env에서
환경변수로 옮긴다.

- 다른 이름(키·토큰 등)은 옮기지 않는다. LangSmith가 필요한 것만 환경변수가 된다.
- 이미 환경변수에 있는 값은 덮어쓰지 않는다. 서버·compose가 준 값이나 쉘에서 내보낸 값이 우선이다.
- LangSmith는 환경변수를 처음 읽을 때 값을 기억한다. 첫 추적 전에 옮겨야 해서 서버 시작 때 한다.
- 로그에는 옮긴 이름만 남기고 값은 남기지 않는다.

개인정보 정책: LangSmith는 외부 회사의 클라우드라서 추적 입력에 든 질문 원문·코멘트(그리고 사용자
ID)가 거기에 저장된다. 그래서 입력·출력을 숨기는 설정(LANGSMITH_HIDE_INPUTS·
LANGSMITH_HIDE_OUTPUTS)은
개발 환경에서만 끌 수 있다. 개발은 APP_ENV를 local·develop·dev·development 중 하나로 직접 정한
환경이다. 그 밖의 환경(운영, APP_ENV를 정하지 않은 곳)에서는 설정에 false가 적혀 있어도 true로
강제하고 경고한다 — 운영에 개발용 설정이 잘못 들어가도 질문·코멘트가 나가지 않게 하려는 것이다.

프로젝트 이름: API 키 하나로 개발·운영 서버가 같은 계정에 올려도 섞이지 않게, LANGSMITH_PROJECT를
정하지 않았으면 APP_ENV로 muro-<환경> 이름을 만든다(local→muro-local, develop→muro-develop,
production·prod→muro-prod). APP_ENV를 직접 정하지 않았으면 muro-unset으로 보내고 경고한다 —
운영과 개발 어느 쪽에도 섞이지 않고 눈에 띄게 하려는 것이다. 직접 정한 이름은 건드리지 않는다.
"""

import logging
import os
import re
from pathlib import Path

from dotenv import dotenv_values

from app.config import Settings

logger = logging.getLogger("muro.observability")

ENV_FILE = Path(".env")  # app.config의 Settings가 읽는 파일과 같다
PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
DEV_ENVS = frozenset({"local", "develop", "dev", "development"})
_TRACING_SWITCHES = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
)
_API_KEYS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_PROJECT_NAMES = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
_HIDE_INPUTS = ("LANGSMITH_HIDE_INPUTS", "LANGCHAIN_HIDE_INPUTS")
_HIDE_OUTPUTS = ("LANGSMITH_HIDE_OUTPUTS", "LANGCHAIN_HIDE_OUTPUTS")


def explicit_app_env(settings: Settings) -> str | None:
    """APP_ENV를 직접 정했으면 그 값, 안 정했으면(기본값 local) None."""
    return settings.app_env if "app_env" in settings.model_fields_set else None


def is_dev_env(settings: Settings) -> bool:
    """APP_ENV를 직접 정했고 개발 값일 때만 True. 안 정했으면(기본값 local) 운영으로 본다."""
    env = explicit_app_env(settings)
    return env is not None and env.strip().lower() in DEV_ENVS


def project_name_for(app_env: str | None) -> str:
    """APP_ENV로 만드는 LangSmith 프로젝트 이름. 정하지 않았으면 muro-unset."""
    if not app_env or not app_env.strip():
        return "muro-unset"
    slug = re.sub(r"[^a-z0-9]+", "-", app_env.strip().lower()).strip("-")
    if not slug:
        return "muro-unset"
    return f"muro-{'prod' if slug == 'production' else slug}"


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
    """추적이 켜져 있는데 API 키가 없으면 서버 로그에 경고한다. 꺼져 있으면 조용하다."""
    if not _is_true(_TRACING_SWITCHES):
        return
    if not any(os.environ.get(name) for name in _API_KEYS):
        logger.warning(
            "LangSmith 추적이 켜져 있지만 LANGSMITH_API_KEY가 없어서 추적이 올라가지 않아요"
        )


def apply_project_name(app_env: str | None) -> None:
    """프로젝트 이름을 정하지 않았으면 APP_ENV로 만든다. 추적이 꺼져 있으면 아무것도 안 한다."""
    if not _is_true(_TRACING_SWITCHES):
        return
    if any(os.environ.get(name) for name in _PROJECT_NAMES):
        return  # 직접 정한 이름이 우선이다
    name = project_name_for(app_env)
    os.environ["LANGSMITH_PROJECT"] = name
    if app_env and app_env.strip():
        logger.info("LangSmith 프로젝트 이름을 APP_ENV(%s)로 만들었어요: %s", app_env, name)
    else:
        logger.warning(
            "LangSmith 프로젝트 이름을 정하지 않았고 APP_ENV도 안 정해서 %s로 보내요. "
            "운영·개발과 섞이지 않게 APP_ENV나 LANGSMITH_PROJECT를 정해 주세요",
            name,
        )


def apply_privacy_policy(*, dev: bool) -> None:
    """운영(개발이 아닌 환경)에서는 입력·출력 숨김을 true로 강제한다. 추적이 꺼져 있으면 안 한다."""
    if not _is_true(_TRACING_SWITCHES):
        return
    hidden = _is_true(_HIDE_INPUTS) and _is_true(_HIDE_OUTPUTS)
    if dev:
        if not hidden:
            logger.info(
                "개발 환경이라 LangSmith 입력·출력을 숨기지 않아요. "
                "질문과 코멘트가 LangSmith로 올라가니 테스트 데이터로만 쓰세요"
            )
        return
    if hidden:
        return
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    logger.warning(
        "LangSmith 입력·출력 숨김이 꺼져 있었지만 개발 환경(APP_ENV를 local·develop·dev로 "
        "직접 설정한 곳)이 아니라서 true로 강제로 켰어요. 질문 원문과 코멘트가 밖으로 나가지 않아요"
    )


def setup_langsmith(
    env_file: Path | str = ENV_FILE, *, dev: bool = False, app_env: str | None = None
) -> None:
    export_langsmith_env(env_file)
    apply_project_name(app_env)
    apply_privacy_policy(dev=dev)
    check_langsmith_setup()
