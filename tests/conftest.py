import os

# 테스트는 항상 가짜 클라이언트·고정 토큰으로 돈다 (app import 전에 설정)
os.environ["AI_CLIENT_MODE"] = "fake"
os.environ["INTERNAL_TOKEN"] = "test-token"

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.dependencies import get_job_repository  # noqa: E402
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


@pytest.fixture
def jobs() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def client(jobs: InMemoryJobRepository) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_job_repository] = lambda: jobs
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
            "genre": "K-Pop",
        },
        "comment": None,
        "createdAt": "2026-08-24T11:14:00Z",
    }
