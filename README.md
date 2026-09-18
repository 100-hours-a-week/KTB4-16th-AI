# Muro AI Server

같은 코드베이스로 프로세스 3개를 띄운다.

| 프로세스 | 실행 | 역할 |
|---|---|---|
| ① AI Gateway | `uvicorn app.main:app --port 8000` | 동기 API + 비동기 작업 접수(202) |
| ② Moderation | `uvicorn app.moderation_main:app --port 8001` | 채팅 모더레이션 |
| ③ Worker | `python -m app.worker_main` | `ai_jobs` 큐 소비 (임베딩·RECAP·클러스터) |

## 모델 서빙 계획

| 역할 | V1 | V2 |
|---|---|---|
| LLM (무드 태깅·리라이팅·요약 등) | Claude API | 미정 |
| 텍스트 임베딩 | OpenAI `text-embedding-3-small` | 로컬 모델, GCP CPU (ONNX INT8) |
| CLIP (사진 임베딩) | Replicate `openai/clip` | RunPod 직접 서빙 |
| 모더레이션 | — | KcELECTRA, GCP CPU (ONNX INT8) |

- V1 필요 키: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `REPLICATE_API_TOKEN`
- 모델 교체는 `app/clients/` 구현체만 바꾼다. 서비스·API 코드는 그대로
- 임베딩 모델을 바꾸면 벡터 차원·좌표계가 달라지므로 새 마이그레이션 + 전체 재임베딩 필요 (`model_versions` 컬럼으로 추적)

## 로컬 실행

```bash
cp .env.example .env                               # AI_CLIENT_MODE=fake면 키 없이 동작
docker compose -f docker-compose.dev.yml up --build
python scripts/send_dummy_locks.py                 # 더미 자물쇠 → /api/embeddings/generate
```

- Swagger: http://localhost:8000/docs
- 모든 `/api/*`(health 제외)는 `X-Internal-Token` 헤더 필요

## 개발

```bash
pip install -r requirements-dev.txt
ruff format . && ruff check .     # CI와 동일
pytest
```

- 의존성은 `requirements.txt`에 추가 (CI가 이 파일로 설치)
- 커밋 메시지는 `feat:` / `fix:` / `chore:` / `test:` / `docs:` — main 릴리스 버전이 여기서 결정됨

## 구조

```
app/
├── api/         HTTP 경계 — 검증·위임만
├── services/    기능별 처리 순서
├── components/  기능 간 재사용 AI 로직 (기능 번호를 모름)
├── clients/     외부 API 어댑터 (fake 구현 포함)
├── db/          커넥션·테이블·리포지토리 (ai_jobs 큐 포함)
├── schemas/     요청·응답 (camelCase)
└── workers/     큐 작업 실행 단위
```

의존 방향: `api → services → components → clients / db`

## 기능4 임베딩 흐름

```
POST /api/embeddings/generate → ai_jobs에 등록 → 202 QUEUED
worker → EmbeddingService
  ├─ photoUrl → CLIP 이미지 벡터
  ├─ track    → LLM 무드 설명 → 텍스트 벡터
  └─ comment  → 텍스트 벡터 (null이면 생략)
  → record_embeddings upsert (recordId 기준 1행)
실패 시 지수 백오프로 재시도, max 초과 시 status=failed
```
