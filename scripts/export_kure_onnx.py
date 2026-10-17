"""KURE-v1(nlpai-lab/KURE-v1)을 서버용 ONNX(fp32)로 변환하고, 원본과 같은 벡터를 내는지 확인한다.

서버(app/components/kure_model.py)는 PyTorch 없이 onnxruntime + tokenizers만 쓴다. 그래서 변환은
이 스크립트를 따로 돌려서 한 번만 하고, 결과 파일을 팀 HF 비공개 저장소에 올려 둔다.
Docker 빌드가 그 저장소에서 파일을 받는다 (Dockerfile의 KURE_MODEL_REPO).

모델은 토큰마다 벡터(last_hidden_state)를 내고, 문장 벡터는 sentence-transformers 설정(1_Pooling:
첫 토큰 CLS, 2_Normalize: 길이 1로 정규화)대로 서버 코드가 만든다. 여기서도 같은 방식으로 만들어
원본(PyTorch)과 비교한다.

실행 (서버 환경과 따로 임시 가상환경에서. torch·transformers는 requirements.txt에 없다):
    pip install torch transformers onnx onnxscript onnxruntime tokenizers huggingface_hub
    python scripts/export_kure_onnx.py --out /tmp/kure_onnx

결과 폴더에 생기는 파일 (모두 HF 저장소의 같은 위치에 올린다):
    kure_v1_fp32.onnx          그래프 (약 3.5MB)
    kure_v1_fp32.onnx.data     가중치 (약 2.3GB) — 위 파일과 같은 폴더에 있어야 읽힌다
    tokenizer.json             토크나이저 (약 17MB)

올리기 (토큰은 터미널에서 직접 로그인하고, 이 스크립트나 채팅에 붙여 넣지 않는다):
    hf auth login
    hf upload <HF 사용자>/<저장소> /tmp/kure_onnx . --private --revision v1
    (--revision v1이 없으면 만들어 준다. Dockerfile의 KURE_MODEL_REPO·KURE_MODEL_REVISION이
    저장소 이름·v1과 같아야 하고, 받는 쪽 토큰(GitHub Secrets의 HF_TOKEN)이
    이 저장소를 읽을 수 있어야 한다)
"""

import argparse
import math
import shutil
from pathlib import Path

MODEL_NAME = "nlpai-lab/KURE-v1"
MODEL_FILE = "kure_v1_fp32.onnx"
MAX_TOKENS = 512  # 서버의 kure_max_tokens와 같은 값으로 확인한다
CHECK_TEXTS = [
    "비 오는 날 퇴근길에 들었던 잔잔한 노래",
    "동탄역 카페거리를 걸으며 노을을 봤다",
    "친구들이랑 신나게 놀던 여름밤",
    "오늘은 아무 일도 없었지만 마음이 편했다",
    "사람 많은 곳은 싫고 조용한 곳이 좋아",
]
MIN_COSINE = 0.9999


def export(out_dir: Path) -> None:
    import torch
    from transformers import AutoModel

    class TokenEmbeddings(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
    ids = torch.randint(5, 1000, (2, 16), dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=2, max=8192)
    with torch.no_grad():
        torch.onnx.export(
            TokenEmbeddings(model).eval(),
            (ids, torch.ones_like(ids)),
            str(out_dir / MODEL_FILE),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
            external_data=True,
            dynamo=True,
        )


def copy_tokenizer(out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    shutil.copy(hf_hub_download(MODEL_NAME, "tokenizer.json"), out_dir / "tokenizer.json")


def verify(out_dir: Path) -> None:
    """서버가 쓰는 코드(OnnxKureModel)와 원본(PyTorch)이 같은 벡터를 내는지 본다."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from app.components.kure_model import load_kure_model

    served = load_kure_model(str(out_dir), MODEL_NAME, dim=1024, threads=2, max_tokens=MAX_TOKENS)
    if served is None:
        raise SystemExit(f"{out_dir}에서 모델 파일을 읽지 못했어요.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    reference = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
    batch = tokenizer(
        CHECK_TEXTS, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
    )
    with torch.no_grad():
        cls = reference(**batch).last_hidden_state[:, 0, :]
    expected = torch.nn.functional.normalize(cls, dim=1).tolist()

    worst = 1.0
    for text, got, want in zip(CHECK_TEXTS, served.embed(CHECK_TEXTS), expected, strict=True):
        cosine = sum(a * b for a, b in zip(got, want, strict=True))
        worst = min(worst, cosine)
        print(f"  코사인 {cosine:.6f}  {text}")
    if not math.isfinite(worst) or worst < MIN_COSINE:
        raise SystemExit(f"원본과 달라요 (최솟값 {worst:.6f} < {MIN_COSINE}). 올리지 마세요.")
    print(f"확인 완료: 원본과의 코사인 최솟값 {worst:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KURE-v1을 ONNX로 변환하고 검증한다")
    parser.add_argument("--out", required=True, type=Path, help="결과를 저장할 폴더")
    parser.add_argument("--verify-only", action="store_true", help="변환 없이 확인만 한다")
    args = parser.parse_args()

    if not args.verify_only:
        export(args.out)
        copy_tokenizer(args.out)
    verify(args.out)


if __name__ == "__main__":
    main()
