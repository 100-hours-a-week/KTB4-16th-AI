"""텍스트 임베딩 교체(OpenAI → KURE-v1 ONNX) — 모델 어댑터, 내부 서비스, 클라이언트, 설정."""

import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app.clients.embedding_client import LocalEmbeddingClient
from app.components.kure_model import BATCH_SIZE, FakeKureModel, OnnxKureModel, load_kure_model
from app.config import SIMILARITY_DEFAULTS, Settings
from app.dependencies import get_kure_model
from app.embedding_main import create_app
from app.exceptions import EmbeddingError
from app.schemas.text_embed import MAX_TEXTS_PER_REQUEST
from tests.conftest import AUTH


class _Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _Tokenizer:
    """글자 수만큼 토큰이 있는 것처럼 흉내 낸다. 배치 안에서는 가장 긴 글에 맞춰 패딩한다."""

    def encode_batch(self, texts):
        return [_Encoding(list(range(1, len(t) + 1))) for t in texts]


class _Session:
    """토큰마다 벡터를 돌려주는 모델 흉내. 첫 토큰(CLS)만 [3, 4, 0]이고 나머지는 다른 값."""

    def __init__(self):
        self.calls: list[int] = []

    def run(self, _outputs, feed):
        batch = len(feed["input_ids"])
        self.calls.append(batch)
        seq = max(len(row) for row in feed["input_ids"])
        token_vectors = [[[3.0, 4.0, 0.0]] + [[9.0, 9.0, 9.0]] * (seq - 1) for _ in range(batch)]
        return [token_vectors]


def _model(session=None):
    return OnnxKureModel(
        session or _Session(), _Tokenizer(), model="kure-test", dim=3, to_array=lambda rows: rows
    )


def test_sentence_vector_is_the_normalized_first_token():
    """sentence-transformers 설정 그대로: 첫 토큰(CLS)만 쓰고, 길이를 1로 맞춘다."""
    [vector] = _model().embed(["안녕"])

    assert all(math.isclose(a, b) for a, b in zip(vector, [0.6, 0.8, 0.0], strict=True))
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)


def test_many_texts_are_split_into_batches_and_keep_their_order():
    session = _Session()

    vectors = _model(session).embed(["가"] * (BATCH_SIZE * 2 + 1))

    assert len(vectors) == BATCH_SIZE * 2 + 1
    assert session.calls == [BATCH_SIZE, BATCH_SIZE, 1]


def test_missing_model_files_load_as_none(tmp_path):
    """빌드 때 HF 토큰이 없어 모델이 없으면 서버는 뜨고 임베딩 요청만 503이다."""
    assert load_kure_model(str(tmp_path), "m", dim=1024, threads=1, max_tokens=512) is None


def test_fake_model_is_deterministic_and_has_the_configured_dim():
    model = FakeKureModel(dim=8)

    first, second = model.embed(["같은 글", "같은 글"])

    assert first == second
    assert len(first) == 8


# ---- 내부 임베딩 서비스 (embedding_main) ----


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_embed_returns_one_vector_per_text(client):
    res = client.post("/api/embed", json={"texts": ["가", "나"]}, headers=AUTH)

    assert res.status_code == 200
    body = res.json()
    assert body["dim"] == 1024
    assert len(body["vectors"]) == 2 and len(body["vectors"][0]) == 1024


def test_embed_needs_the_internal_token(client):
    assert client.post("/api/embed", json={"texts": ["가"]}).status_code == 401


def test_embed_rejects_blank_and_too_many_texts(client):
    blank = client.post("/api/embed", json={"texts": ["가", "  "]}, headers=AUTH)
    many = client.post(
        "/api/embed", json={"texts": ["가"] * (MAX_TEXTS_PER_REQUEST + 1)}, headers=AUTH
    )
    empty = client.post("/api/embed", json={"texts": []}, headers=AUTH)

    assert blank.status_code == many.status_code == empty.status_code == 400


def test_missing_model_is_503_but_health_still_answers():
    app = create_app()
    app.dependency_overrides[get_kure_model] = lambda: None
    test_client = TestClient(app)

    res = test_client.post("/api/embed", json={"texts": ["가"]}, headers=AUTH)
    health = test_client.get("/api/embed/health")

    assert res.status_code == 503 and res.json()["code"] == "MODEL_UNAVAILABLE"
    assert health.status_code == 200 and health.json()["model"] is None


def test_health_shows_model_and_needs_no_token(client):
    res = client.get("/api/embed/health")

    assert res.status_code == 200
    assert res.json()["model"] == "nlpai-lab/KURE-v1" and res.json()["dim"] == 1024


# ---- gateway·worker가 쓰는 클라이언트 ----


def _client(handler) -> LocalEmbeddingClient:
    return LocalEmbeddingClient(
        base_url="http://embedding",
        token="test-token",
        model="m",
        dim=2,
        timeout=5,
        transport=httpx.MockTransport(handler),
    )


async def test_client_sends_token_and_returns_vectors():
    seen = {}

    def handler(request):
        seen["token"] = request.headers["X-Internal-Token"]
        seen["path"] = request.url.path
        return httpx.Response(200, json={"vectors": [[1.0, 0.0]]})

    vectors = await _client(handler).embed(["가"])

    assert vectors == [[1.0, 0.0]]
    assert seen == {"token": "test-token", "path": "/api/embed"}


async def test_client_splits_long_lists_into_requests_the_service_accepts():
    sizes = []

    def handler(request):
        import json

        texts = json.loads(request.content)["texts"]
        sizes.append(len(texts))
        return httpx.Response(200, json={"vectors": [[0.0, 1.0]] * len(texts)})

    vectors = await _client(handler).embed(["가"] * (MAX_TEXTS_PER_REQUEST + 5))

    assert len(vectors) == MAX_TEXTS_PER_REQUEST + 5
    assert sizes == [MAX_TEXTS_PER_REQUEST, 5]


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(503, json={"code": "MODEL_UNAVAILABLE"}),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"other": 1}),
        lambda request: httpx.Response(200, json={"vectors": []}),  # 개수가 다름
        lambda request: httpx.Response(200, json={"vectors": [[1.0, 0.0, 0.0]]}),  # 차원이 다름
    ],
    ids=["status", "not-json", "no-vectors-key", "count-mismatch", "dim-mismatch"],
)
async def test_client_turns_any_bad_answer_into_embedding_error(handler):
    with pytest.raises(EmbeddingError):
        await _client(handler).embed(["가"])


async def test_client_turns_connection_failure_into_embedding_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbeddingError):
        await _client(handler).embed(["가"])


# ---- 설정: 모델마다 다른 차원·기준값 ----


def test_kure_is_the_default_provider():
    s = Settings(_env_file=None)

    assert s.text_embedding_provider == "kure"
    assert s.text_vector_dim == 1024
    assert s.memory_min_similarity == s.personal_similarity == SIMILARITY_DEFAULTS["kure"]


def test_openai_provider_keeps_its_own_dim_and_threshold():
    s = Settings(_env_file=None, text_embedding_provider="openai")

    assert s.text_vector_dim == 1536
    assert s.memory_min_similarity == s.personal_similarity == 0.40


def test_threshold_override_wins_over_the_provider_default():
    s = Settings(_env_file=None, memory_search_min_similarity=0.5, personal_similarity_min=0.7)

    assert s.memory_min_similarity == 0.5
    assert s.personal_similarity == 0.7


def test_old_openai_env_values_do_not_change_kure_settings():
    """배포 환경에 TEXT_EMBEDDING_MODEL/DIM(OpenAI 값)이 남아 있어도 KURE 설정은 그대로다."""
    s = Settings(
        _env_file=None, text_embedding_model="text-embedding-3-small", text_embedding_dim=1536
    )

    assert s.kure_model_name == "nlpai-lab/KURE-v1" and s.text_vector_dim == 1024
