"""기능6 유해성 판정 모델 — KcELECTRA 파인튜닝 모델을 ONNX로 바꿔 CPU에서 추론한다.

서버에는 PyTorch 없이 onnxruntime + tokenizers만 쓴다.
모델 파일은 비공개 HF 저장소(dionypark/mulo-moderation-kcelectra)에서 Docker 빌드 때 받는다.
학습·평가 기록: 개발/유해성검증모델실험/05_실험결과.md
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

MAX_LENGTH = 128


class ModerationModel(Protocol):
    version: str

    def toxic_probability(self, text: str) -> float: ...


class OnnxModerationModel:
    def __init__(self, model_dir: Path, version: str, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_dir / "model_fp32.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_LENGTH)
        labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self._toxic_index = int(next(k for k, v in labels.items() if v == "toxic"))
        self.version = version

    def toxic_probability(self, text: str) -> float:
        enc = self._tokenizer.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        logits = self._session.run(None, feed)[0][0]
        exp = np.exp(logits - logits.max())
        return float(exp[self._toxic_index] / exp.sum())


class FakeModerationModel:
    """키·모델 파일 없이 로컬과 테스트에서 쓰는 가짜 모델. 몇 개 단어만 유해로 본다."""

    version = "fake"
    _words = ("꺼져", "시발", "병신", "정상없다")

    def toxic_probability(self, text: str) -> float:
        return 0.95 if any(w in text for w in self._words) else 0.05


def load_onnx_model(model_dir: str, version: str, threads: int) -> OnnxModerationModel | None:
    """모델 파일이 없으면 None (빌드 때 HF 토큰이 없었던 경우). 서버는 뜨고 판정만 503."""
    path = Path(model_dir)
    if not (path / "model_fp32.onnx").exists():
        return None
    return OnnxModerationModel(path, version=version, threads=threads)
