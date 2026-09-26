# 멀티 스테이지 빌드일때

# 1. 빌드 스테이지
FROM python:3.11 AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# 2. 런타임 스테이지
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1 
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r app && useradd -r -g app appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . .

USER appuser

EXPOSE 8000 8001

# 마이그레이션 → ① gateway(8000) ② moderation(8001) ③ worker 를 한 번에 띄운다 (app/launcher.py)
CMD ["python", "-m", "app.main"]
