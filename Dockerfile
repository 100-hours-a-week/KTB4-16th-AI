FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .

EXPOSE 8000

# 같은 이미지로 command만 바꿔 ① gateway ② moderation ③ worker를 띄운다
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
