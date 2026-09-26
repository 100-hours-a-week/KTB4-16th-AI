FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .

EXPOSE 8000 8001

# 마이그레이션 → ① gateway(8000) ② moderation(8001) ③ worker 를 한 번에 띄운다 (app/launcher.py)
# 따로 띄우고 싶으면 command만 바꾸면 된다: uvicorn app.main:app / uvicorn app.moderation_main:app
# / python -m app.worker_main
CMD ["python", "-m", "app.main"]
