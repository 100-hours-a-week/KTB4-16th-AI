"""환경변수·외부 API 키 설정."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 유사도 기준값의 제공자별 기본값. 모델마다 점수 분포가 달라서 같은 값을 쓸 수 없다.
# - openai: 2026-09-26 더미 자물쇠(서로 다른 코멘트 18개)로 잰 값
# - kure: 2026-10-15 합성 시험셋(코멘트 100개, 질문 30개)에서 OpenAI 0.40과 비슷하거나 나은 지점
#   (응답 76% / 오답 38%). 합성 데이터라 임시값이다 — 실제 코멘트가 쌓이면 다시 맞출 것.
SIMILARITY_DEFAULTS = {"openai": 0.40, "kure": 0.65}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"

    # 백엔드 → AI 서버 내부 인증 (X-Internal-Token)
    internal_token: str = "dev-internal-token"

    # AI 전용 PostgreSQL (pgvector)
    database_url: str = "postgresql+asyncpg://muro:muro@localhost:5432/muro_ai"

    # 백엔드 MySQL (읽기 전용) — 기능2 RECAP용 records 조회
    backend_mysql_url: str = ""

    # 백엔드 콜백 (AI → 백엔드, RECAP 완료 알림) — 경로·인증 방식 확정 전 추측값
    backend_internal_url: str = ""

    # fake: 키 없이 결정적 가짜 벡터·텍스트로 동작 (로컬 개발·테스트용)
    # real: 실제 외부 API 호출
    ai_client_mode: Literal["fake", "real"] = "fake"

    # 모델 서빙 계획 (README 참고)
    #   V1: 전부 외부 API — Anthropic(LLM), OpenAI(텍스트 임베딩), Replicate(CLIP)
    #   V2: GCP CPU(ONNX) — KcELECTRA, 텍스트 임베딩 / RunPod — CLIP, 로컬 LLM(general 작업만)

    # LLM — 작업 성격에 따라 두 갈래로 나눈다
    #   knowledge: 곡·가수 지식이 필요한 작업 (음악 무드 묘사, 투표 문항 생성) → 프론티어 유지
    #   general:   말 바꾸기만 하는 작업 (검색어 리라이팅, 요약, 답변, 클러스터 이름) → V2에 로컬
    llm_knowledge_provider: Literal["anthropic", "local"] = "anthropic"
    llm_general_provider: Literal["anthropic", "local"] = "anthropic"

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # 로컬 LLM (V2, RunPod) — 구현 예정
    local_llm_url: str = ""
    local_llm_model: str = ""

    # 텍스트 임베딩 (코멘트·음악 무드 텍스트·검색 질문). V2부터 KURE-v1(자체 서빙)이 기본이다.
    # "openai"는 예전 방식(1536차원)이라, 벡터 칸이 1024차원인 지금은 되돌리기용으로만 남겨 둔다
    # (마이그레이션 0010을 되돌리고 다시 임베딩해야 쓸 수 있음).
    text_embedding_provider: Literal["kure", "openai"] = "kure"
    openai_api_key: str = ""
    text_embedding_model: str = "text-embedding-3-small"  # provider=openai일 때만 쓴다
    text_embedding_dim: int = 1536  # provider=openai일 때만 쓴다

    # KURE-v1 (nlpai-lab/KURE-v1, ONNX fp32). 컨테이너 안의 별도 프로세스(embedding_main)가 모델을
    # 한 번만 올리고, gateway·worker는 HTTP로 부른다. 모델 파일은 Docker 빌드 때 HF 비공개
    # 저장소에서 받아 둔다. 배포 환경변수 TEXT_EMBEDDING_MODEL/DIM과 이름을 분리해서, 예전
    # OpenAI 값이 남아 있어도 덮어쓰이지 않게 했다.
    kure_model_name: str = "nlpai-lab/KURE-v1"
    kure_embedding_dim: int = 1024
    kure_model_dir: str = "/app/models/kure"
    kure_threads: int = 2
    kure_max_tokens: int = 512  # 입력 길이 상한. 긴 글 하나가 처리 시간을 크게 늘리지 않게 자른다
    embedding_service_url: str = "http://127.0.0.1:8002"

    # 기능5 기억 검색 (1단계 기준선) — 아래 값은 모두 시험지로 정할 초기값이다.
    # 임베딩 모델을 바꾸면 유사도 값의 분포가 달라지므로 그때 다시 정한다.
    # None이면 임베딩 제공자별 기본값(SIMILARITY_DEFAULTS)을 쓴다. 값을 주면 그 값을 쓴다.
    memory_search_min_similarity: float | None = None
    # 기능1이 "비슷한 과거 코멘트"로 인정하는 최소 유사도. None이면 제공자별 기본값
    personal_similarity_min: float | None = None
    # 곡을 고를 때 질문 벡터에 내 취향 벡터를 섞는 비율 (0=끔, 0.3=상황 70% + 취향 30%).
    # 정답을 알 데이터가 아직 없어서 감으로 둔 초기값 — 같은 상황 세트로 비교해 정한다.
    personal_taste_weight: float = 0.3
    memory_search_comment_weight: float = 0.5
    memory_search_music_weight: float = 0.5
    memory_search_candidates: int = 20  # 검색기마다 가져오는 후보 수
    # 질문에 조건(장소·계절·날씨·기분 등)이 있을 때의 점수 — 마찬가지로 시험지로 정할 초기값
    memory_search_condition_weight: float = 0.5  # 최종 점수에서 조건 점수의 몫 (나머지는 벡터)
    memory_search_location_weight: float = 2.0  # 조건 점수 안에서 장소·지역·동 조건의 가중치
    memory_search_other_condition_weight: float = 1.0  # 그 밖의 조건
    memory_search_scan_limit: int = 5000  # 조건 비교를 위해 읽는 자물쇠의 최대 개수 (최근순)
    memory_search_condition_candidates: int = 50  # 조건으로만 후보가 되는 기록의 최대 개수
    # 프론트는 앞 5건을 먼저 보여주고 "더보기"에서 나머지를 보여준다. 모두 맞는 기록이 이 건수보다
    # 적을 때만 일부만 맞는 기록을 문장으로 제안한다 (요청 limit이 더 작으면 그 값)
    memory_search_display_count: int = 5
    # 제안(일부만 맞는 자물쇠) 토글을 열면 보이는 최대 건수
    memory_search_suggestion_count: int = 3
    # 답변 앞머리 문장(질문한 조건 + 결과 유무, 건수 없음)을 LLM이 쓴다. 꺼 두면 템플릿 문장.
    # LLM에는 조건 이름과 결과 유무만 보낸다 (질문 원문·곡·장소·코멘트·건수는 보내지 않음)
    memory_answer_llm_enabled: bool = False
    memory_answer_llm_timeout_seconds: float = 8.0  # 넘기면 템플릿 문장으로 대신한다

    # CLIP 이미지 임베딩 — Replicate openai/clip (clip-vit-large-patch14)
    # V2에 RunPod 직접 서빙으로 교체 예정. 모델이 바뀌면 차원 변경 → 전체 재임베딩
    replicate_api_token: str = ""
    clip_model: str = "openai/clip"
    clip_embedding_dim: int = 768

    # Spotify — 앱 인증(검색용, 무제한)과 서비스 계정(플레이리스트 저장, B안) 분리
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    # scripts/spotify_service_account_setup.py로 1회 발급. 뮤로 팀 계정 1개 전용
    spotify_service_refresh_token: str = ""

    # 기능1·3 추천 5곡 중 곡 풀에서 채울 곡 수. 0이면 예전처럼 LLM 곡만.
    # 곡 풀을 쓰면 LLM에게는 song_pool_llm_count곡만 받는다 (Spotify 탈락 대비 여유분 포함)
    song_pool_count: int = 3
    song_pool_llm_count: int = 4
    # 곡 풀 몫 중 몇 곡을 "사용자들이 저장한 곡"(track_moods)에서 가져올지. 나머지는 TJ 곡 풀.
    # 1로 둔 이유: 오픈 직후엔 저장된 곡이 적어서 다 거기서 뽑으면 같은 곡만 돌 수 있다.
    # 저장 곡이 쌓이면 올려 가며 실측한다. 0이면 TJ 곡 풀만 사용.
    song_pool_saved_count: int = 1
    # 곡 풀 파일 — Docker 빌드 때 비공개 HF 저장소에서 받아 둔다(공개 레포에 두지 않음).
    # 로컬에서 쓰려면 HF에서 받아 SONG_POOL_PATH로 지정
    song_pool_path: str = "/app/models/song_pool/song_pool.json"

    # 기능6 모더레이션 모델 (KcELECTRA ONNX) — Docker 빌드 때 HF 비공개 저장소에서 받아 둔다
    moderation_model_dir: str = "/app/models/moderation"
    moderation_model_version: str = "kcelectra-v1-stage3"
    moderation_threshold: float = 0.5
    moderation_threads: int = 2

    # 기능6 오늘의 질문 — 최근 며칠치 자물쇠로 장르별 통계를 낼지, 최대 몇 건까지 볼지
    balance_stats_days: int = 90
    balance_stats_limit: int = 5000
    # 기능6 매일 배치 (KST) — 시작 시각에 다음 날 몫을 계산하고, 마감까지 재시도해 끝낸다.
    # 백엔드는 마감 이후에 뷰를 읽는다 (be_user_music_genres, be_vote_questions)
    balance_daily_start: str = "23:00"
    balance_daily_deadline: str = "23:40"
    # 그룹 인원이 이보다 적은 장르의 사용자는 공통 그룹으로 (매칭할 사람이 모이지 않으므로).
    # 1:1 매칭이라 2명부터 그룹 유지(10/15 결정), 오픈 후 매칭 성공률을 보고 조정한다
    balance_min_group_size: int = 2
    # V2까지는 모든 사용자를 공통 그룹으로 묶고 공통 질문만 만든다 (백엔드와 합의, 채팅을 먼저
    # 열기 위해). 장르별 그룹으로 돌아가려면 False — 그때 balance_min_group_size가 쓰인다.
    balance_common_only: bool = True
    # 그룹마다 미리 채워 둘 날짜 수 (배치가 하루 실패해도 질문이 비지 않게)
    balance_question_days_ahead: int = 2

    # 외부 API 공통
    external_timeout_seconds: float = 20.0
    external_max_retries: int = 2

    # python -m app.main (전체 실행)이 띄우는 두 API 서버 포트
    gateway_port: int = 8000
    moderation_port: int = 8001
    embedding_port: int = 8002  # 컨테이너 안(127.0.0.1)에서만 열림

    # Prometheus 지표 (GET /metrics). 인증 없이 열리므로 외부에 노출되지 않는 내부망에서만
    # 가져가야 한다. 인프라 사정으로 꺼야 할 때 METRICS_ENABLED=false
    metrics_enabled: bool = True

    # 워커 (Postgres 기반 작업 큐)
    # 블루그린 대기 VM은 false — 워커는 LB를 안 거치고 공유 DB 큐에서 직접 작업을 가져가서,
    # 대기 VM에서도 돌면 다른 버전 코드가 운영 작업을 처리해 버린다. CD가 트래픽 전환 때 바꾼다.
    run_worker: bool = True
    worker_poll_interval_seconds: float = 1.0
    job_max_attempts: int = 3
    job_stale_after_seconds: int = 600

    @property
    def text_vector_dim(self) -> int:
        """텍스트 임베딩 벡터의 차원 — DB 벡터 칸 크기와 같아야 한다 (제공자에 따라 다름)."""
        if self.text_embedding_provider == "kure":
            return self.kure_embedding_dim
        return self.text_embedding_dim

    @property
    def memory_min_similarity(self) -> float:
        if self.memory_search_min_similarity is not None:
            return self.memory_search_min_similarity
        return SIMILARITY_DEFAULTS[self.text_embedding_provider]

    @property
    def personal_similarity(self) -> float:
        if self.personal_similarity_min is not None:
            return self.personal_similarity_min
        return SIMILARITY_DEFAULTS[self.text_embedding_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
