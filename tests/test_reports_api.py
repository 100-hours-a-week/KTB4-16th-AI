import pytest

from tests.conftest import AUTH

BATCH = "/api/reports/batch-generate"
REPORT_JOB = "report"
NOTIFY_JOB = "report_batch_notify"


def _report_job_keys(jobs) -> list[str]:
    return sorted(key for (job_type, key) in jobs.jobs if job_type == REPORT_JOB)


def test_monthly_report_not_found_is_404(client):
    res = client.get("/api/reports/12?year=2026&month=9", headers=AUTH)

    assert res.status_code == 404
    assert res.json()["code"] == "MONTHLY_REPORT_NOT_FOUND"


async def test_monthly_report_returns_saved_content(client, monthly_reports):
    await monthly_reports.save(
        user_id=12,
        year=2026,
        month=9,
        ai_recap_text="당신은 주로 카페에서 잔잔한 음악을 들으며 편안한 시간을 보냈어요.",
        photo_scenes=[
            {"tag": "카페", "count": 6, "ratio": 60},
            {"tag": "노을", "count": 4, "ratio": 40},
        ],
    )

    res = client.get("/api/reports/12?year=2026&month=9", headers=AUTH)

    assert res.status_code == 200
    assert res.json() == {
        "userId": 12,
        "year": 2026,
        "month": 9,
        "photoScenes": [
            {"tag": "카페", "count": 6, "ratio": 60},
            {"tag": "노을", "count": 4, "ratio": 40},
        ],
        "aiRecap": {
            "status": "COMPLETED",
            "text": "당신은 주로 카페에서 잔잔한 음악을 들으며 편안한 시간을 보냈어요.",
        },
    }


def test_monthly_report_needs_token(client):
    res = client.get("/api/reports/12?year=2026&month=9")

    assert res.status_code == 401


def test_batch_generate_keys_jobs_by_batch_request_id(client, jobs):
    res = client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [1, 2], "batchRequestId": "req-A"},
        headers=AUTH,
    )

    assert res.status_code == 202
    assert res.json() == {
        "status": "QUEUED",
        "jobId": "report_batch_2026-9",  # 형태는 예전 그대로
        "targetCount": 2,
        "batchRequestId": "req-A",
    }
    assert _report_job_keys(jobs) == ["req-A:1", "req-A:2"]
    assert jobs.jobs[(REPORT_JOB, "req-A:1")]["payload"] == {
        "user_id": 1,
        "year": 2026,
        "month": 9,
        "batch_request_id": "req-A",
    }


def test_batch_generate_without_batch_request_id_keeps_year_month_behavior(client, jobs):
    """구버전 백엔드 요청: 키·응답 모양이 기존과 같다."""
    res = client.post(BATCH, json={"year": 2026, "month": 9, "userIds": [1]}, headers=AUTH)

    assert res.status_code == 202
    assert res.json() == {"status": "QUEUED", "jobId": "report_batch_2026-9", "targetCount": 1}
    assert _report_job_keys(jobs) == ["2026-9-1"]


def test_same_batch_request_id_is_not_enqueued_twice(client, jobs):
    body = {"year": 2026, "month": 9, "userIds": [1, 2], "batchRequestId": "req-A"}
    client.post(BATCH, json=body, headers=AUTH)
    # 첫 접수 뒤 일이 진행됐다고 가정 — 재요청이 이 상태를 pending으로 되돌리면 안 된다
    jobs.jobs[(REPORT_JOB, "req-A:1")]["status"] = "done"

    res = client.post(BATCH, json=body, headers=AUTH)

    assert res.status_code == 202
    assert res.json()["batchRequestId"] == "req-A"
    assert res.json()["targetCount"] == 2
    assert jobs.jobs[(REPORT_JOB, "req-A:1")]["status"] == "done"
    assert _report_job_keys(jobs) == ["req-A:1", "req-A:2"]


def test_different_batch_request_ids_for_same_month_are_independent(client, jobs):
    """같은 달 수동 트리거를 2번 보내도 ID가 다르면 서로 다른 배치다."""
    client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [1, 2], "batchRequestId": "req-A"},
        headers=AUTH,
    )
    client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [1], "batchRequestId": "req-B"},
        headers=AUTH,
    )

    assert _report_job_keys(jobs) == ["req-A:1", "req-A:2", "req-B:1"]


def test_legacy_request_releases_previous_notify_marker(client, jobs):
    """ID 없는 재생성 요청은 지난 배치의 알림 전송권을 돌려놓는다(기존 동작 유지)."""
    jobs.jobs[(NOTIFY_JOB, "2026-9")] = {"id": 99, "payload": {}, "status": "done"}

    client.post(BATCH, json={"year": 2026, "month": 9, "userIds": [1]}, headers=AUTH)

    assert (NOTIFY_JOB, "2026-9") not in jobs.jobs


def test_request_with_batch_request_id_does_not_touch_year_month_notify_marker(client, jobs):
    jobs.jobs[(NOTIFY_JOB, "2026-9")] = {"id": 99, "payload": {}, "status": "done"}

    client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [1], "batchRequestId": "req-A"},
        headers=AUTH,
    )

    assert (NOTIFY_JOB, "2026-9") in jobs.jobs


def test_batch_with_no_target_users_enqueues_nothing(client, jobs):
    res = client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [], "batchRequestId": "req-A"},
        headers=AUTH,
    )

    assert res.status_code == 202
    assert res.json()["targetCount"] == 0
    assert _report_job_keys(jobs) == []


@pytest.mark.parametrize(
    "bad_id",
    ["", "has space", "bad:id", "under_score", "한글", "a" * 65],
)
def test_invalid_batch_request_id_is_rejected(client, jobs, bad_id):
    res = client.post(
        BATCH,
        json={"year": 2026, "month": 9, "userIds": [1], "batchRequestId": bad_id},
        headers=AUTH,
    )

    assert res.status_code == 400
    assert _report_job_keys(jobs) == []
