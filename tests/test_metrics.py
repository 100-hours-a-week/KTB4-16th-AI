"""Prometheus 지표(GET /metrics) 테스트."""

import re

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def metrics_client() -> TestClient:
    return TestClient(create_app())


def _counted(body: str, handler: str, method: str, status: str) -> float | None:
    """http_requests_total에서 해당 경로·메서드·상태 묶음의 값을 읽는다."""
    pattern = rf'http_requests_total\{{[^}}]*handler="{re.escape(handler)}"[^}}]*\}} ([0-9.e+]+)'
    for line in body.splitlines():
        if (
            line.startswith("http_requests_total{")
            and f'handler="{handler}"' in line
            and f'method="{method}"' in line
            and f'status="{status}"' in line
        ):
            return float(re.search(pattern, line).group(1))
    return None


def test_metrics_endpoint_is_open_and_returns_prometheus_text(metrics_client):
    res = metrics_client.get("/metrics")  # 내부 토큰 없이 연다

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert "# HELP http_requests_total" in res.text
    # 파이썬 기본 지표도 같이 나간다 (프로세스 CPU·메모리 지표는 리눅스 전용이라 확인 안 함)
    assert "python_info" in res.text
    assert "python_gc_objects_collected_total" in res.text


def test_requests_are_counted_by_route_status_and_method(metrics_client):
    # 내부 토큰이 없어서 401 — 그래도 요청은 센다
    metrics_client.post("/api/memory-search", json={"userId": 7, "queryText": "비"})

    body = metrics_client.get("/metrics").text

    assert _counted(body, "/api/memory-search", "POST", "4xx") == 1.0


def test_path_parameters_are_recorded_as_templates_not_real_ids(metrics_client):
    metrics_client.delete("/api/embeddings/12345")

    body = metrics_client.get("/metrics").text

    assert 'handler="/api/embeddings/{recordId}"' in body
    assert "12345" not in body  # 실제 ID가 지표에 남지 않는다


def test_unknown_paths_are_grouped_into_one_series(metrics_client):
    metrics_client.get("/no/such/path/777")
    metrics_client.get("/another/unknown/888")

    body = metrics_client.get("/metrics").text

    assert _counted(body, "none", "GET", "4xx") == 2.0
    assert "777" not in body and "888" not in body


def test_health_checks_and_metrics_reads_are_not_counted(metrics_client):
    metrics_client.get("/health")
    metrics_client.get("/api/health")
    metrics_client.get("/metrics")

    body = metrics_client.get("/metrics").text

    for path in ("/health", "/api/health", "/metrics"):
        assert f'handler="{path}"' not in body


def test_metrics_is_not_listed_in_openapi_docs(metrics_client):
    paths = metrics_client.get("/openapi.json").json()["paths"]

    assert "/metrics" not in paths


def test_creating_the_app_twice_does_not_clash_and_keeps_counts_separate():
    first = TestClient(create_app())
    second = TestClient(create_app())  # 기본 저장소를 공유하면 여기서 지표 이름 중복 오류가 난다
    first.post("/api/memory-search", json={"userId": 7, "queryText": "비"})

    assert _counted(first.get("/metrics").text, "/api/memory-search", "POST", "4xx") == 1.0
    assert _counted(second.get("/metrics").text, "/api/memory-search", "POST", "4xx") is None


def test_metrics_can_be_turned_off_with_an_environment_variable(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        assert client.get("/metrics").status_code == 404
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
