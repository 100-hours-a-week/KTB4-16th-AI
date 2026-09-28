from tests.conftest import AUTH


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
        photo_scenes=["카페", "노을"],
    )

    res = client.get("/api/reports/12?year=2026&month=9", headers=AUTH)

    assert res.status_code == 200
    assert res.json() == {
        "userId": 12,
        "year": 2026,
        "month": 9,
        "photoScenes": ["카페", "노을"],
        "aiRecap": {
            "status": "COMPLETED",
            "text": "당신은 주로 카페에서 잔잔한 음악을 들으며 편안한 시간을 보냈어요.",
        },
    }


def test_monthly_report_needs_token(client):
    res = client.get("/api/reports/12?year=2026&month=9")

    assert res.status_code == 401
