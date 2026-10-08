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

# 마이그레이션 → ① gateway(8000) ② moderation(8001) ③ worker 를 한 번에 띄운다 (app/launcher.py)
CMD ["python", "-m", "app.main"]
