from app.components.question_generator import QuestionGenerator
from app.db.repositories.balance_stats_repository import GenreStats
from app.dependencies import get_balance_service
from app.services.balance_service import BalanceService
from tests.conftest import AUTH
from tests.test_song_curator import ScriptedLLM


class FixedStats:
    def __init__(self, stats=None, broken=False):
        self.stats, self.broken, self.asked = stats or GenreStats.empty(), broken, None

    async def get(self, music_genre):
        self.asked = music_genre
        if self.broken:
            raise RuntimeError("DB 장애")
        return self.stats


def _override(client, stats, response):
    client.app.dependency_overrides[get_balance_service] = lambda: BalanceService(
        stats, QuestionGenerator(ScriptedLLM(response))
    )


def test_generate_questions(client):
    stats = FixedStats()
    _override(client, stats, "인디는 어디서? | 한강 | 방\n밤에 듣는 인디는? | 잔잔한 | 신나는")
    res = client.post(
        "/api/balance/questions/generate", json={"musicGenre": "인디음악", "count": 2}, headers=AUTH
    )
    assert res.status_code == 200
    assert res.json() == {
        "questions": [
            {
                "question": "인디는 어디서?",
                "optionA": "한강",
                "optionB": "방",
                "musicGenre": "인디음악",
            },
            {
                "question": "밤에 듣는 인디는?",
                "optionA": "잔잔한",
                "optionB": "신나는",
                "musicGenre": "인디음악",
            },
        ]
    }
    assert stats.asked == "인디음악"


def test_unknown_genre_is_400(client):
    res = client.post(
        "/api/balance/questions/generate", json={"musicGenre": "시티팝"}, headers=AUTH
    )
    assert res.status_code == 400
    assert res.json()["field"] == "musicGenre"


def test_stats_failure_still_generates(client):
    _override(client, FixedStats(broken=True), "오늘 기분엔? | 발라드 | 댄스")
    res = client.post("/api/balance/questions/generate", json={"musicGenre": "공통"}, headers=AUTH)
    assert res.status_code == 200
    assert len(res.json()["questions"]) == 1


def test_no_valid_question_is_503(client):
    _override(client, FixedStats(), "형식이 틀린 답")
    res = client.post("/api/balance/questions/generate", json={"musicGenre": "공통"}, headers=AUTH)
    assert res.status_code == 503


def test_cluster_endpoints_are_removed(client):
    assert client.post("/api/balance/cluster-assign", json={}, headers=AUTH).status_code == 404
    assert client.post("/api/balance/clusters/rebuild", headers=AUTH).status_code == 404
