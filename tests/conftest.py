import os

# 테스트는 항상 가짜 클라이언트·고정 토큰으로 돈다 (app import 전에 설정)
os.environ["AI_CLIENT_MODE"] = "fake"
os.environ["INTERNAL_TOKEN"] = "test-token"

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.dependencies import get_embedding_store, get_job_repository  # noqa: E402
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


class InMemoryEmbeddingStore:
    def __init__(self) -> None:
        self.rows: dict[int, Any] = {}

    async def upsert(self, data) -> None:
        self.rows[data.record_id] = data

    async def delete(self, record_id: int) -> None:
        self.rows.pop(record_id, None)


@pytest.fixture
def jobs() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def store() -> InMemoryEmbeddingStore:
    return InMemoryEmbeddingStore()


@pytest.fixture
def client(jobs: InMemoryJobRepository, store: InMemoryEmbeddingStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_job_repository] = lambda: jobs
    app.dependency_overrides[get_embedding_store] = lambda: store
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
