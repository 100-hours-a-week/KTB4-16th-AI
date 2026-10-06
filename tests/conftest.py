import os

# 테스트는 항상 가짜 클라이언트·고정 토큰으로 돈다 (app import 전에 설정)
os.environ["AI_CLIENT_MODE"] = "fake"
os.environ["INTERNAL_TOKEN"] = "test-token"

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.repositories.track_lookup_repository import LookupEntry  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_embedding_store,
    get_job_repository,
    get_monthly_report_repository,
)
from app.main import create_app  # noqa: E402

AUTH = {"X-Internal-Token": "test-token"}


class InMemoryJobRepository:
    def __init__(self) -> None:
        self.jobs: dict[tuple[str, str], dict[str, Any]] = {}

    async def enqueue(
        self, *, job_type: str, dedupe_key: str, payload: dict[str, Any], max_attempts: int
    ) -> int:
        key = (job_type, dedupe_key)
        job_id = self.jobs[key]["id"] if key in self.jobs else len(self.jobs) + 1
        self.jobs[key] = {"id": job_id, "payload": payload, "status": "pending"}
        return job_id

    async def cancel(self, *, job_type: str, dedupe_key: str) -> None:
        self.jobs.pop((job_type, dedupe_key), None)

    async def release_once(self, *, job_type: str, dedupe_key: str) -> None:
        self.jobs.pop((job_type, dedupe_key), None)


class InMemoryLookups:
    """track_lookups 대역 — broken=True면 DB 장애처럼 예외를 던진다."""

    def __init__(self, entries: list[LookupEntry] | None = None, broken: bool = False):
        self.rows = {e.lookup_key: e for e in entries or []}
        self.broken = broken

    async def get_many(self, keys):
        if self.broken:
            raise RuntimeError("DB 장애")
        return {k: self.rows[k] for k in keys if k in self.rows}

    async def save_many(self, entries):
        if self.broken:
            raise RuntimeError("DB 장애")
        self.rows |= {e.lookup_key: e for e in entries}


class InMemoryEmbeddingStore:
    def __init__(self) -> None:
        self.rows: dict[int, Any] = {}

    async def upsert(self, data) -> None:
        self.rows[data.record_id] = data

    async def delete(self, record_id: int) -> None:
        self.rows.pop(record_id, None)


class InMemoryMonthlyReportRepository:
    def __init__(self) -> None:
        self.rows: dict[tuple[int, int, int], dict[str, Any]] = {}

    async def save(
        self,
        *,
        user_id: int,
        year: int,
        month: int,
        ai_recap_text: str,
        photo_scenes: list[dict[str, Any]],
    ) -> None:
        self.rows[(user_id, year, month)] = {
            "ai_recap_text": ai_recap_text,
            "photo_scenes": photo_scenes,
        }

    async def get(self, *, user_id: int, year: int, month: int) -> Any:
        from app.db.repositories.report_repository import MonthlyReportRecord

        row = self.rows.get((user_id, year, month))
        return None if row is None else MonthlyReportRecord(**row)

    async def get_many(self, *, user_ids: list[int], year: int, month: int) -> dict[int, Any]:
        from app.db.repositories.report_repository import MonthlyReportRecord

        return {
            uid: MonthlyReportRecord(**row)
            for uid in user_ids
            if (row := self.rows.get((uid, year, month))) is not None
        }


@pytest.fixture
def jobs() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def store() -> InMemoryEmbeddingStore:
    return InMemoryEmbeddingStore()


@pytest.fixture
def monthly_reports() -> InMemoryMonthlyReportRepository:
    return InMemoryMonthlyReportRepository()


@pytest.fixture
def client(
    jobs: InMemoryJobRepository,
    store: InMemoryEmbeddingStore,
    monthly_reports: InMemoryMonthlyReportRepository,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_job_repository] = lambda: jobs
    app.dependency_overrides[get_embedding_store] = lambda: store
    app.dependency_overrides[get_monthly_report_repository] = lambda: monthly_reports
    return TestClient(app)


@pytest.fixture
def record_payload() -> dict[str, Any]:
    return {
        "recordId": 1029,
        "userId": 1044,
        "photoUrl": "https://cdn.muro.app/photos/1029.jpg",
        "track": {
            "externalTrackId": "6rqhFgbbKwnb9MLmUQDhG6",
            "title": "밤편지",
            "artistName": "아이유",
        },
        "comment": None,
        "createdAt": "2026-08-24T11:14:00Z",
    }
