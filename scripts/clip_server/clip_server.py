"""RunPod Pod에서 CLIP ViT-L/14(768차원)를 서빙하는 작은 서버 — Replicate openai/clip의 대용품.

서비스의 ReplicateClipClient와 같은 모양으로 쓰도록, 입력은 {"image": URL 또는 data URI} /
{"text": 글}, 출력은 {"embedding": [...]}이다. 글 여러 개는 {"texts": [...]}를 보내면
{"embeddings": [...]}가 나온다. 임베딩은 모델이 내는 값 그대로(정규화하지 않음)다.
서비스는 코사인 유사도를 쓰므로 크기는 무관하다.

- 인증: 요청 헤더 Authorization: Bearer <CLIP_API_KEY>. 키는 서버를 켤 때 환경변수로 넣는다.
- 이미지 주소(URL)는 이 서버가 직접 내려받는다. 키를 가진 호출자만 쓸 수 있고,
  10MB를 넘으면 거절한다.
- 가중치는 이미 받아 둔 Hugging Face 캐시(HF_HOME)에서만 읽는다(인터넷으로 새로 받지 않는다).

실행 (Pod 웹 터미널, 이 파일을 /workspace/clip_server.py로 저장한 뒤):
    export CLIP_API_KEY='<직접 만든 비밀번호>'
    nohup python3 /workspace/clip_server.py > /workspace/clip_server.log 2>&1 &
선택 환경변수: CLIP_MODEL(기본 openai/clip-vit-large-patch14), CLIP_DTYPE(float16·float32,
기본 float16), CLIP_PORT(기본 8001).
"""

import base64
import binascii
import hmac
import io
import logging
import os
import sys
import threading
from typing import Protocol

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("muro.clip_server")

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TEXTS = 64
MAX_TEXT_CHARS = 1000
FETCH_TIMEOUT_SECONDS = 20.0


class Encoder(Protocol):
    model_name: str
    dim: int
    device: str

    def encode_image(self, data: bytes) -> list[float]: ...

    def encode_texts(self, texts: list[str]) -> list[list[float]]: ...


class ImageRequest(BaseModel):
    image: str  # http(s) 주소 또는 data:image/...;base64,... 형식


class TextRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None


def load_image_bytes(source: str) -> bytes:
    """data URI는 풀어서, http(s) 주소는 내려받아서 이미지 바이트를 돌려준다."""
    if source.startswith("data:"):
        _, _, payload = source.partition(",")
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(status_code=400, detail="data URI를 읽을 수 없어요.") from e
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="이미지가 너무 커요.")
        return data

    if source.startswith(("http://", "https://")):
        data = bytearray()
        try:
            with httpx.stream(
                "GET", source, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True
            ) as res:
                res.raise_for_status()
                for chunk in res.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise HTTPException(status_code=413, detail="이미지가 너무 커요.")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail="이미지 주소를 내려받지 못했어요.") from e
        return bytes(data)

    raise HTTPException(status_code=400, detail="image는 http(s) 주소나 data URI여야 해요.")


def create_app(encoder: Encoder, api_key: str) -> FastAPI:
    app = FastAPI(title="muro-clip-server")

    def require_key(authorization: str | None = Header(default=None)) -> None:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="키가 맞지 않아요.")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": encoder.model_name,
            "dim": encoder.dim,
            "device": encoder.device,
        }

    @app.post("/encode_image", dependencies=[Depends(require_key)])
    def encode_image(req: ImageRequest) -> dict:
        data = load_image_bytes(req.image)
        try:
            return {"embedding": encoder.encode_image(data)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/encode_text", dependencies=[Depends(require_key)])
    def encode_text(req: TextRequest) -> dict:
        if (req.text is None) == (req.texts is None):
            raise HTTPException(status_code=400, detail="text와 texts 중 하나만 보내세요.")
        texts = [req.text] if req.text is not None else req.texts
        if not texts or len(texts) > MAX_TEXTS or any(len(t) > MAX_TEXT_CHARS for t in texts):
            raise HTTPException(
                status_code=400,
                detail=f"글은 1~{MAX_TEXTS}개, 각 {MAX_TEXT_CHARS}자 이하여야 해요.",
            )
        vectors = encoder.encode_texts(texts)
        if req.text is not None:
            return {"embedding": vectors[0]}
        return {"embeddings": vectors}

    return app


class ClipEncoder:
    """transformers CLIPModel로 이미지·글을 임베딩한다. torch는 여기서 처음 불러온다."""

    def __init__(self, model_name: str, dtype_name: str):
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

        self._torch = torch
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
        # 모델 호출은 한 번에 하나씩 — GPU를 같이 쓰는 vLLM에 부담을 덜 주려는 것이다
        self._lock = threading.Lock()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        try:  # OpenAI 원본(PIL 보간) 전처리에 더 가까운 느린 처리기를 먼저 쓴다
            self._image_processor = AutoImageProcessor.from_pretrained(
                model_name, local_files_only=True, use_fast=False
            )
        except Exception:
            logger.warning("느린 이미지 처리기를 못 불러서 기본 처리기를 써요.", exc_info=True)
            self._image_processor = AutoImageProcessor.from_pretrained(
                model_name, local_files_only=True
            )
        self._model = (
            CLIPModel.from_pretrained(model_name, local_files_only=True, dtype=self._dtype)
            .to(self.device)
            .eval()
        )
        self.dim = int(self._model.config.projection_dim)
        self._warm_up()

    def _warm_up(self) -> None:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, format="PNG")
        self.encode_image(buffer.getvalue())
        self.encode_texts(["warm up"])

    def encode_image(self, data: bytes) -> list[float]:
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError("이미지를 읽을 수 없어요.") from e
        pixel_values = self._image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=self._dtype)
        with self._lock, self._torch.inference_mode():
            pooled = self._model.vision_model(pixel_values=pixel_values).pooler_output
            embedding = self._model.visual_projection(pooled)
        return embedding[0].float().cpu().tolist()

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        tokens = self._tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        with self._lock, self._torch.inference_mode():
            pooled = self._model.text_model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            ).pooler_output
            embeddings = self._model.text_projection(pooled)
        return embeddings.float().cpu().tolist()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ.get("CLIP_API_KEY", "")
    if not api_key:
        sys.exit("CLIP_API_KEY 환경변수가 필요해요. export CLIP_API_KEY='...' 후 다시 실행하세요.")
    model_name = os.environ.get("CLIP_MODEL", DEFAULT_MODEL)
    dtype_name = os.environ.get("CLIP_DTYPE", "float16")
    if dtype_name not in ("float16", "float32"):
        sys.exit("CLIP_DTYPE은 float16 또는 float32여야 해요.")
    port = int(os.environ.get("CLIP_PORT", "8001"))

    encoder = ClipEncoder(model_name, dtype_name)
    logger.info("CLIP 준비 완료: %s dim=%d device=%s", model_name, encoder.dim, encoder.device)

    import uvicorn

    uvicorn.run(create_app(encoder, api_key), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
