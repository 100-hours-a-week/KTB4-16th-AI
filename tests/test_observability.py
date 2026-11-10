import logging
import os

import pytest

from app.observability import check_langsmith_setup, export_langsmith_env, setup_langsmith


@pytest.fixture(autouse=True)
def isolated_environ(monkeypatch):
    """환경변수를 이 테스트 안에서만 바꾼다 (끝나면 원래대로)."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in list(os.environ):
        if name.startswith(("LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delitem(os.environ, name)


def _env_file(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_langsmith_values_in_the_env_file_become_environment_variables(tmp_path):
    path = _env_file(
        tmp_path,
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=lsv2_secret\nLANGSMITH_PROJECT=muro\n",
    )

    moved = export_langsmith_env(path)

    assert moved == ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "muro"


def test_only_langsmith_and_langchain_names_are_moved_so_other_secrets_stay_in_the_file(tmp_path):
    path = _env_file(
        tmp_path,
        "MURO_TEST_OTHER_SECRET=sk-secret\nMURO_TEST_OTHER_TOKEN=tok\nLANGCHAIN_TRACING_V2=true\n",
    )

    moved = export_langsmith_env(path)

    assert moved == ["LANGCHAIN_TRACING_V2"]
    assert "MURO_TEST_OTHER_SECRET" not in os.environ
    assert "MURO_TEST_OTHER_TOKEN" not in os.environ


def test_a_value_that_is_already_an_environment_variable_wins_over_the_env_file(tmp_path):
    os.environ["LANGSMITH_TRACING"] = "false"  # 서버·compose·쉘이 준 값
    path = _env_file(tmp_path, "LANGSMITH_TRACING=true\nLANGSMITH_PROJECT=muro\n")

    moved = export_langsmith_env(path)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert moved == ["LANGSMITH_PROJECT"]


def test_a_missing_env_file_is_fine(tmp_path):
    assert export_langsmith_env(tmp_path / "없는파일.env") == []


def test_quotes_and_comments_in_the_env_file_are_handled(tmp_path):
    path = _env_file(
        tmp_path,
        '# LangSmith\nLANGSMITH_PROJECT="muro memory"  # 이름\nLANGSMITH_HIDE_INPUTS=true\n',
    )

    export_langsmith_env(path)

    assert os.environ["LANGSMITH_PROJECT"] == "muro memory"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"


def test_the_log_lists_names_but_never_values(tmp_path, caplog):
    path = _env_file(tmp_path, "LANGSMITH_API_KEY=lsv2_very_secret_value\nLANGSMITH_TRACING=true\n")

    with caplog.at_level(logging.INFO, logger="muro.observability"):
        export_langsmith_env(path)

    assert "LANGSMITH_API_KEY" in caplog.text
    assert "lsv2_very_secret_value" not in caplog.text


def test_tracing_works_once_the_values_are_moved_before_langsmith_reads_them(tmp_path):
    """.env에만 적어 둔 값이 실제로 LangSmith가 읽는 환경변수가 된다 (네트워크 호출 없음)."""
    from langsmith import utils

    utils.get_env_var.cache_clear()  # 첫 읽기 값을 기억하는 캐시를 비워 처음 읽는 상황으로 만든다
    try:
        export_langsmith_env(_env_file(tmp_path, "LANGSMITH_TRACING=true\n"))
        assert utils.tracing_is_enabled() is True
    finally:
        utils.get_env_var.cache_clear()


def test_warns_when_tracing_is_on_without_an_api_key(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        check_langsmith_setup()

    assert "LANGSMITH_API_KEY가 없어서" in caplog.text


def test_warns_when_tracing_is_on_but_inputs_are_not_hidden(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "lsv2_x"

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        check_langsmith_setup()

    assert "LANGSMITH_HIDE_INPUTS" in caplog.text and "질문 원문" in caplog.text


def test_no_warning_when_tracing_is_off_or_everything_is_set(caplog):
    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        check_langsmith_setup()  # 꺼져 있으면 조용하다
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "lsv2_x"
        os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
        check_langsmith_setup()  # 구버전 이름(LANGCHAIN_TRACING_V2)으로 켜도 설정이 맞으면 조용하다

    assert caplog.text == ""


def test_setup_langsmith_moves_the_values_and_checks_them(tmp_path, caplog):
    path = _env_file(tmp_path, "LANGSMITH_TRACING=true\n")  # 키·숨김 설정이 빠진 .env

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        setup_langsmith(path)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert "LANGSMITH_API_KEY가 없어서" in caplog.text
    assert "LANGSMITH_HIDE_INPUTS" in caplog.text
