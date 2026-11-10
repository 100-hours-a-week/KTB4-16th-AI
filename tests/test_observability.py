import logging
import os

import pytest

from app.config import Settings
from app.observability import (
    apply_privacy_policy,
    apply_project_name,
    check_langsmith_setup,
    explicit_app_env,
    export_langsmith_env,
    is_dev_env,
    project_name_for,
    setup_langsmith,
)


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

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        check_langsmith_setup()

    assert "LANGSMITH_API_KEY가 없어서" in caplog.text


def test_no_key_warning_when_tracing_is_off_or_the_key_is_set(caplog):
    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        check_langsmith_setup()  # 꺼져 있으면 조용하다
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 구버전 이름으로 켜도 키가 있으면 조용하다
        os.environ["LANGSMITH_API_KEY"] = "lsv2_x"
        check_langsmith_setup()

    assert caplog.text == ""


# ---- 개인정보 정책: 입력·출력 숨김은 개발에서만 끌 수 있다 ------------------------------------


def test_outside_dev_the_hide_settings_are_forced_on_even_if_the_file_says_false(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "false"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "false"

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        apply_privacy_policy(dev=False)

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert "강제로 켰어요" in caplog.text


def test_outside_dev_hiding_is_forced_even_when_the_hide_settings_are_missing(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"

    apply_privacy_policy(dev=False)

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"


def test_outside_dev_nothing_is_said_when_everything_is_already_hidden(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"

    with caplog.at_level(logging.INFO, logger="muro.observability"):
        apply_privacy_policy(dev=False)

    assert caplog.text == ""


def test_in_dev_visible_inputs_are_allowed_and_only_noted_not_warned(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_HIDE_INPUTS"] = "false"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "false"

    with caplog.at_level(logging.INFO, logger="muro.observability"):
        apply_privacy_policy(dev=True)

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "false"  # 바꾸지 않는다
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "false"
    assert "개발 환경이라" in caplog.text
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_when_tracing_is_off_the_policy_changes_nothing():
    apply_privacy_policy(dev=False)

    assert "LANGSMITH_HIDE_INPUTS" not in os.environ
    assert "LANGSMITH_HIDE_OUTPUTS" not in os.environ


def test_app_env_counts_as_dev_only_when_set_explicitly_to_a_dev_value(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    assert (
        is_dev_env(Settings(_env_file=None)) is False
    )  # 정하지 않음(기본값 local) → 운영으로 본다
    for value in ("local", "develop", "dev", "development", "DEV"):
        assert is_dev_env(Settings(_env_file=None, app_env=value)) is True, value
    for value in ("prod", "production", "staging", ""):
        assert is_dev_env(Settings(_env_file=None, app_env=value)) is False, value


def test_setup_moves_the_values_then_forces_hiding_outside_dev(tmp_path, caplog):
    # 개발 PC에서 쓰던 .env(키 없음, 숨김 꺼짐)가 운영에 잘못 들어간 상황
    path = _env_file(
        tmp_path,
        "LANGSMITH_TRACING=true\nLANGSMITH_HIDE_INPUTS=false\nLANGSMITH_HIDE_OUTPUTS=false\n",
    )

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        setup_langsmith(path, dev=False)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert "강제로 켰어요" in caplog.text and "LANGSMITH_API_KEY가 없어서" in caplog.text


def test_setup_in_dev_keeps_the_visible_settings_from_the_file(tmp_path):
    path = _env_file(
        tmp_path,
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=lsv2_x\n"
        "LANGSMITH_HIDE_INPUTS=false\nLANGSMITH_HIDE_OUTPUTS=false\n",
    )

    setup_langsmith(path, dev=True)

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "false"


# ---- 프로젝트 이름: LANGSMITH_PROJECT를 안 정했으면 APP_ENV로 만든다 ----------------------


@pytest.mark.parametrize(
    ("app_env", "name"),
    [
        ("local", "muro-local"),
        ("develop", "muro-develop"),
        (" DEV ", "muro-dev"),
        ("production", "muro-prod"),
        ("prod", "muro-prod"),
        ("Stage 2!", "muro-stage-2"),
        (None, "muro-unset"),
        ("", "muro-unset"),
        ("  ", "muro-unset"),
        ("!!!", "muro-unset"),
    ],
)
def test_project_name_comes_from_app_env(app_env, name):
    assert project_name_for(app_env) == name


def test_dev_and_prod_get_different_project_names_so_one_api_key_does_not_mix_them():
    assert project_name_for("develop") != project_name_for("production")
    assert project_name_for("unset-env-placeholder") != project_name_for(None)


def test_the_project_name_is_filled_in_when_tracing_is_on_and_none_is_set(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"

    with caplog.at_level(logging.INFO, logger="muro.observability"):
        apply_project_name("develop")

    assert os.environ["LANGSMITH_PROJECT"] == "muro-develop"
    assert "muro-develop" in caplog.text
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_a_project_name_set_by_hand_is_never_replaced():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "내가-정한-이름"

    apply_project_name("production")

    assert os.environ["LANGSMITH_PROJECT"] == "내가-정한-이름"


def test_the_old_langchain_project_name_also_counts_as_set_by_hand():
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "옛-이름"

    apply_project_name("production")

    assert "LANGSMITH_PROJECT" not in os.environ


def test_no_project_name_is_made_when_tracing_is_off():
    apply_project_name("production")

    assert "LANGSMITH_PROJECT" not in os.environ


def test_without_app_env_the_project_is_muro_unset_and_it_is_warned(caplog):
    os.environ["LANGSMITH_TRACING"] = "true"

    with caplog.at_level(logging.WARNING, logger="muro.observability"):
        apply_project_name(None)

    assert os.environ["LANGSMITH_PROJECT"] == "muro-unset"
    assert "APP_ENV" in caplog.text and "muro-unset" in caplog.text


def test_app_env_is_none_unless_it_was_set_explicitly(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)

    assert explicit_app_env(Settings(_env_file=None)) is None
    assert explicit_app_env(Settings(_env_file=None, app_env="develop")) == "develop"


def test_setup_names_the_project_from_app_env_but_keeps_a_name_written_in_the_file(tmp_path):
    named = _env_file(tmp_path, "LANGSMITH_TRACING=true\nLANGSMITH_PROJECT=파일에-적은-이름\n")
    setup_langsmith(named, dev=True, app_env="develop")
    assert os.environ["LANGSMITH_PROJECT"] == "파일에-적은-이름"

    del os.environ["LANGSMITH_PROJECT"]
    unnamed = _env_file(tmp_path, "LANGSMITH_TRACING=true\n")
    setup_langsmith(unnamed, dev=True, app_env="develop")
    assert os.environ["LANGSMITH_PROJECT"] == "muro-develop"
