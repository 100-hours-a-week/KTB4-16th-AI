"""KURE-v1 텍스트 임베딩 추론 (ONNX). 서버에는 PyTorch 없이 onnxruntime + tokenizers만 쓴다.

모델은 토큰마다 벡터(last_hidden_state)를 내고, 문장 벡터는 sentence-transformers 설정
(1_Pooling: 첫 토큰 CLS, 2_Normalize: 길이 1로 정규화)대로 여기서 만든다. ONNX 변환본이 원본과
같은 벡터를 내는지는 변환 때 확인했다(코사인 최솟값 0.999999, scripts/export_kure_onnx.py 참고).

gateway·worker는 이 모델을 직접 올리지 않고 embedding_main(별도 프로세스)을 HTTP로 부른다.
그래서 모델이 메모리에 한 번만 올라간다.
"""

import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from app.clients.fake import fake_vector

MODEL_FILE = "kure_v1_fp32.onnx"  # 같은 폴더의 kure_v1_fp32.onnx.data(가중치)를 함께 읽는다
TOKENIZER_FILE = "tokenizer.json"
BATCH_SIZE = 16  # 한 번에 모델에 넣는 문장 수. 크면 패딩과 메모리가 늘어난다


class TextEmbeddingModel(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _numpy_int64(rows: list[list[int]]) -> Any:
    import numpy as np

    return np.array(rows, dtype=np.int64)


class OnnxKureModel:
    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        *,
        model: str,
        dim: int,
        to_array: Callable[[list[list[int]]], Any] = _numpy_int64,
    ):
        self.model = model
        self.dim = dim
        self._session = session
        self._tokenizer = tokenizer
        self._to_array = to_array
        # 모델 호출을 한 번에 하나씩만 한다. 요청이 겹쳐도 CPU·메모리가 한꺼번에 늘지 않는다
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors += self._embed_batch(texts[start : start + BATCH_SIZE])
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)
        feed = {
            "input_ids": self._to_array([e.ids for e in encodings]),
            "attention_mask": self._to_array([e.attention_mask for e in encodings]),
        }
        with self._lock:
            hidden = self._session.run(None, feed)[0]
        # 1_Pooling: 각 문장의 첫 토큰(CLS) 벡터. 2_Normalize: 길이를 1로
        return [_normalize([float(x) for x in row[0]]) for row in hidden]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class FakeKureModel:
    """모델 파일이 없는 로컬·테스트 환경용. 같은 글이면 같은 벡터를 낸다."""

    def __init__(self, dim: int, model: str = "fake-kure"):
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(f"text:{t}", self.dim) for t in texts]


def load_kure_model(
    model_dir: str, model_name: str, dim: int, threads: int, max_tokens: int
) -> OnnxKureModel | None:
    """모델 파일이 없으면 None (서버는 뜨고 요청만 503). 있으면 한 번 돌려 차원을 확인한다."""
    path = Path(model_dir)
    if not (path / MODEL_FILE).exists() or not (path / TOKENIZER_FILE).exists():
        return None

    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(path / MODEL_FILE), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(path / TOKENIZER_FILE))
    tokenizer.enable_truncation(max_length=max_tokens)
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    model = OnnxKureModel(session, tokenizer, model=model_name, dim=dim)
    got = len(model.embed(["차원 확인"])[0])
    if got != dim:
        raise RuntimeError(f"KURE 모델 출력 차원({got})이 설정값({dim})과 달라요.")
    return model
