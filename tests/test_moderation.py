import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.components.moderation_model import FakeModerationModel, load_onnx_model
from app.dependencies import get_moderation_service
from app.moderation_main import create_app
from app.services.moderation_service import ModerationService
from tests.conftest import AUTH


def _body(text: str) -> dict:
    return {"requestId": "req_1", "roomId": "room_1", "userId": "u_1", "text": text}


@pytest.fixture
def mod_client() -> TestClient:
    return TestClient(create_app())


def test_toxic_message(mod_client):
    body = _body("국힙 듣는 애들 중에 정상없다")
    res = mod_client.post("/api/moderation/check", json=body, headers=AUTH)
    assert res.status_code == 200
    assert res.json() == {
        "requestId": "req_1",
        "isToxic": True,
        "confidence": 0.95,
        "category": None,
        "modelVersion": "fake",
    }


def test_clean_message(mod_client):
    res = mod_client.post("/api/moderation/check", json=_body("보컬 고음 죽인다"), headers=AUTH)
    assert res.status_code == 200
    assert res.json()["isToxic"] is False
    assert res.json()["confidence"] == 0.05


def test_needs_internal_token(mod_client):
    res = mod_client.post("/api/moderation/check", json=_body("안녕"))
    assert res.status_code == 401


def test_empty_text_is_400(mod_client):
    res = mod_client.post("/api/moderation/check", json=_body(""), headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "text"


def test_missing_model_is_503():
    # 빌드 때 HF 토큰이 없어 모델 파일이 없는 경우: 서버는 뜨고 판정만 503
    app = create_app()
    app.dependency_overrides[get_moderation_service] = lambda: ModerationService(None, 0.5)
    res = TestClient(app).post("/api/moderation/check", json=_body("안녕"), headers=AUTH)
    assert res.status_code == 503
    assert res.json()["code"] == "MODEL_UNAVAILABLE"


def test_threshold_is_applied():
    app = create_app()
    app.dependency_overrides[get_moderation_service] = lambda: ModerationService(
        FakeModerationModel(), threshold=0.99
    )
    res = TestClient(app).post("/api/moderation/check", json=_body("꺼져"), headers=AUTH)
    assert res.json()["isToxic"] is False


def test_health_shows_model_version(mod_client):
    res = mod_client.get("/api/moderation/health")
    assert res.status_code == 200
    assert res.json()["modelVersion"] == "fake"


def test_missing_model_files_load_as_none(tmp_path):
    assert load_onnx_model(str(tmp_path), "v", threads=1) is None


# 실제 모델 파일이 있을 때만 (로컬 확인용): MODERATION_MODEL_TEST_DIR=<model_fp32.onnx 있는 폴더>
_real_dir = os.environ.get("MODERATION_MODEL_TEST_DIR", "")


@pytest.mark.skipif(not (_real_dir and Path(_real_dir).exists()), reason="모델 파일 없음")
def test_real_onnx_model():
    model = load_onnx_model(_real_dir, "kcelectra-v2-stage7", threads=2)
    assert model is not None
    assert model.toxic_probability("국힙 듣는 애들 중에 정상없다") > 0.9
    assert model.toxic_probability("국힙 개좋아 이번 앨범 찢었다") < 0.1


class _AlwaysToxic:
    """규칙이 모델을 건너뛰었는지 보려고, 부르면 무조건 유해로 답하는 대역."""

    version = "always-toxic"

    def toxic_probability(self, text: str) -> float:
        return 0.99


def _check_with(model, text):
    app = create_app()
    app.dependency_overrides[get_moderation_service] = lambda: ModerationService(model, 0.5)
    return TestClient(app).post("/api/moderation/check", json=_body(text), headers=AUTH)


SHORT_INPUTS = [
    "ㅇ",
    "ㅇㅇ",
    "ㅇㅇ?",
    "ㅋㅋㅋㅋ",
    "ㅠㅠ",
    "?",
    "...",
    "👍",
    "네",
    "ㄱㄱㅋㅋ",
    " ㅎㅎ ",
]
STILL_CHECKED = ["ㅗ", "ㅗㅗ", "ㅄ", "ㄷㅊ", "ㄸㄹㅇ", "ㄱㅅㄲ", "ㅅㅂ", "잘자", "꺼져"]


@pytest.mark.parametrize("text", SHORT_INPUTS)
def test_short_inputs_skip_the_model(text):
    res = _check_with(_AlwaysToxic(), text)

    assert res.json()["isToxic"] is False
    assert res.json()["confidence"] == 0.0


@pytest.mark.parametrize("text", STILL_CHECKED)
def test_swear_jamo_and_real_words_still_go_to_the_model(text):
    assert _check_with(_AlwaysToxic(), text).json()["isToxic"] is True


def test_short_input_passes_even_without_model_files():
    res = _check_with(None, "ㅋㅋ")

    assert res.status_code == 200
    assert res.json()["modelVersion"] == "short-text-rule"


def test_default_threshold_is_0_6():
    from app.config import get_settings

    assert get_settings().moderation_threshold == 0.6
