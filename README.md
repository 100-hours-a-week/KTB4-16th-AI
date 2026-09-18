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
| LLM — knowledge (음악 무드 묘사, 투표 문항 생성) | Claude API | Claude API 유지 |
| LLM — general (검색어 리라이팅, 요약, 답변, 클러스터 이름) | Claude API | RunPod 로컬 LLM |
| 텍스트 임베딩 | OpenAI `text-embedding-3-small` | ④ Embedding Service, GCP CPU (ONNX) |
| CLIP (사진 임베딩) | Replicate `openai/clip` | RunPod 직접 서빙 |
| 모더레이션 | — | ② Moderation Service, KcELECTRA, GCP CPU (ONNX) |

- V1 필요 키: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `REPLICATE_API_TOKEN`
- LLM은 곡 지식이 필요한 작업(knowledge)과 언어 변환만 하는 작업(general)으로 나눈다. 로컬 LLM은 곡을 잘 몰라 지어낼 위험이 있어서, knowledge는 프론티어에 남긴다 (`LLM_KNOWLEDGE_PROVIDER` / `LLM_GENERAL_PROVIDER`)
- 모델 교체는 `app/clients/` 구현체만 바꾼다. 서비스·API 코드는 그대로
- 정밀도(fp32/INT8)는 모델마다 실측으로 정한다. KcELECTRA는 INT8+ONNX에서 recall이 12.5%p 떨어져 fp32+ONNX 채택 (위키 2단계)
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
