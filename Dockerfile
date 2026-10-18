# syntax=docker/dockerfile:1
# 멀티 스테이지 빌드일때

# 1. 빌드 스테이지
FROM python:3.11 AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 기능6 모더레이션 모델(ONNX)을 비공개 HF 저장소에서 받는다.
# 토큰은 빌드 시크릿으로만 넘겨서 이미지 레이어에 남지 않는다 (--secret id=hf_token).
# 토큰이 없으면 모델 없이 빌드되고, 서버는 뜨지만 모더레이션 판정만 503을 낸다.
ARG MODERATION_MODEL_REPO=dionypark/mulo-moderation-kcelectra
ARG MODERATION_MODEL_REVISION=v1-stage3
RUN --mount=type=secret,id=hf_token \
    mkdir -p /models/moderation && \
    if [ -s /run/secrets/hf_token ]; then \
      for f in model_fp32.onnx tokenizer.json labels.json; do \
        curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token)" \
          -o "/models/moderation/$f" \
          "https://huggingface.co/${MODERATION_MODEL_REPO}/resolve/${MODERATION_MODEL_REVISION}/$f"; \
      done; \
    else \
      echo "hf_token 없음: 모더레이션 모델 없이 빌드"; \
    fi

# 기능3 곡 풀(곡별 장르·무드·설명)도 비공개 HF 데이터 저장소에서 받는다 — 공개 레포에 두지 않는다.
# 곡 풀을 고치면 HF에 새 태그(pool-v2 ...)를 올리고 아래 태그를 바꿔 커밋한다.
# 없으면 곡 풀 없이 빌드되고, 사진 추천은 LLM 곡만으로 동작한다.
ARG SONG_POOL_REPO=dionypark/mulo-song-pool
ARG SONG_POOL_REVISION=pool-v3
RUN --mount=type=secret,id=hf_token \
    mkdir -p /models/song_pool && \
    if [ -s /run/secrets/hf_token ]; then \
      curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token)" \
        -o /models/song_pool/song_pool.json \
        "https://huggingface.co/datasets/${SONG_POOL_REPO}/resolve/${SONG_POOL_REVISION}/song_pool.json"; \
    else \
      echo "hf_token 없음: 곡 풀 없이 빌드"; \
    fi

# 텍스트 임베딩 모델 KURE-v1(ONNX fp32)을 HF 비공개 저장소에서 받는다. 가중치가 약 2.3GB라 이미지가 그만큼 커진다.
# 모델 파일은 scripts/export_kure_onnx.py로 만들어 올려 둔 것이다.
# 이 저장소는 모더레이션·곡 풀과 계정이 달라서 별도 읽기 토큰(빌드 시크릿 hf_token_kure)으로 받는다.
# 마이그레이션 0010이 기존 벡터를 비우는 배포라 모델 없는 이미지를 올리면 안 된다.
# 그래서 시크릿이 없거나 받기에 하나라도 실패하면 빌드를 멈춘다.
ARG KURE_MODEL_REPO=wonyyyyyyyyyyy/muro-kure-v1-onnx
ARG KURE_MODEL_REVISION=v1
RUN --mount=type=secret,id=hf_token_kure \
    set -eu; \
    mkdir -p /models/kure; \
    if [ ! -s /run/secrets/hf_token_kure ]; then \
      echo "hf_token_kure 없음: KURE 모델 없이는 빌드할 수 없음" >&2; exit 1; \
    fi; \
    for f in kure_v1_fp32.onnx kure_v1_fp32.onnx.data tokenizer.json; do \
      curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token_kure)" \
        -o "/models/kure/$f" \
        "https://huggingface.co/${KURE_MODEL_REPO}/resolve/${KURE_MODEL_REVISION}/$f"; \
    done


# 2. 런타임 스테이지
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r app && useradd -r -g app appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /app/models
COPY . .

USER appuser

EXPOSE 8000 8001

# 마이그레이션 → ① gateway(8000) ② moderation(8001) ③ embedding(8002, 컨테이너 안에서만) ④ worker 를
# 한 번에 띄운다 (app/launcher.py)
CMD ["python", "-m", "app.main"]
