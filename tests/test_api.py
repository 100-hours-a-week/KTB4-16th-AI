from tests.conftest import AUTH


def test_health_needs_no_token(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert "uptimeSeconds" in res.json()


def test_missing_token_is_401(client, record_payload):
    res = client.post("/api/embeddings/generate", json=record_payload)
    assert res.status_code == 401
    assert res.json() == {
        "code": "UNAUTHORIZED",
        "message": "내부 인증 토큰이 없거나 올바르지 않아요.",
        "field": None,
    }


def test_embeddings_generate_queues_job(client, jobs, record_payload):
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.status_code == 202
    assert res.json() == {"status": "QUEUED", "jobId": "emb_1"}
    assert jobs.jobs[("embedding", "1029")]["payload"]["recordId"] == 1029


def test_same_record_is_deduplicated(client, jobs, record_payload):
    client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.json()["jobId"] == "emb_1"
    assert len(jobs.jobs) == 1


def test_missing_field_is_400_with_camel_field(client, record_payload):
    del record_payload["photoUrl"]
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.status_code == 400
    assert res.json()["code"] == "INVALID_INPUT"
    assert res.json()["field"] == "photoUrl"


def test_unimplemented_feature_is_501(client):
    res = client.post(
        "/api/memory-search", json={"userId": "u_1", "queryText": "비 오던 날"}, headers=AUTH
    )
    assert res.status_code == 501
    assert res.json()["code"] == "NOT_IMPLEMENTED"


def test_missing_photo_is_400(client, record_payload):
    record_payload["photoUrl"] = None
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "photoUrl"


def test_string_record_id_is_400(client, record_payload):
    record_payload["recordId"] = "rec_1029"
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "recordId"


def test_llm_split_uses_separate_clients(monkeypatch):
    from app import dependencies
    from app.clients.anthropic_client import AnthropicClient
    from app.clients.llm_client import LocalLLMClient
    from app.config import get_settings

    monkeypatch.setenv("AI_CLIENT_MODE", "real")
    monkeypatch.setenv("LLM_KNOWLEDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_GENERAL_PROVIDER", "local")
    get_settings.cache_clear()
    dependencies.get_clients.cache_clear()
    try:
        clients = dependencies.get_clients()
        assert isinstance(clients.llm_knowledge, AnthropicClient)
        assert isinstance(clients.llm_general, LocalLLMClient)
    finally:
        get_settings.cache_clear()
        dependencies.get_clients.cache_clear()


def test_delete_removes_pending_job_and_vectors(client, jobs, store, record_payload):
    client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    store.rows[1029] = object()
    res = client.delete("/api/embeddings/1029", headers=AUTH)
    assert res.status_code == 204
    assert ("embedding", "1029") not in jobs.jobs
    assert 1029 not in store.rows


def test_delete_unknown_record_is_still_204(client):
    assert client.delete("/api/embeddings/999", headers=AUTH).status_code == 204


def test_delete_invalid_id_is_400(client):
    res = client.delete("/api/embeddings/abc", headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "recordId"


def test_delete_needs_token(client):
    assert client.delete("/api/embeddings/1029").status_code == 401


def test_missing_track_id_is_400(client, record_payload):
    del record_payload["track"]["externalTrackId"]
    res = client.post("/api/embeddings/generate", json=record_payload, headers=AUTH)
    assert res.status_code == 400
    assert res.json()["field"] == "track.externalTrackId"


def test_root_health_for_deploy_smoke_test(client):
    """CD 스모크 테스트가 curl http://localhost:8000/health 로 확인한다."""
    res = client.get("/health")

    assert res.status_code == 200
    assert res.json()["status"] == "ok"
