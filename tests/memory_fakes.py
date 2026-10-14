"""기능5 기억 검색 테스트용 가짜 구현 — 벡터 검색, 백엔드 상세 조회, 임베더."""

from datetime import datetime

from app.components.vector_search import SimilarRecord
from app.db.repositories.memory_record_repository import MemoryRecord


def hit(record_id: int, similarity: float) -> SimilarRecord:
    return SimilarRecord(record_id=record_id, external_track_id=None, similarity=similarity)


def make_record(record_id: int, **overrides) -> MemoryRecord:
    values = {
        "record_id": record_id,
        "created_at": datetime(2026, 8, 24, 18, 30),
        "title": "밤편지",
        "artist_name": "아이유",
        "external_track_id": "2uwDoSLzTVVmWjsmbo8jhX",
        "album_image_url": "https://img.example/cover.jpg",
        "external_url": "https://open.spotify.com/track/2uwDoSLzTVVmWjsmbo8jhX",
        "place_id": 10,
        "place_name": "오산동",
    }
    values.update(overrides)
    return MemoryRecord(**values)


class FakeVectorSearch:
    """미리 정한 결과를 돌려주고, 어떤 사용자·개수로 불렀는지 기록한다."""

    def __init__(self, comment=None, music=None):
        self.comment = comment or []
        self.music = music or []
        self.calls: list[tuple[str, int, int]] = []

    async def similar_by_comment(self, user_id, query_vector, limit):
        self.calls.append(("comment", user_id, limit))
        return self.comment

    async def similar_by_music(self, user_id, query_vector, limit):
        self.calls.append(("music", user_id, limit))
        return self.music


class FakeMemoryRecords:
    """백엔드 MySQL 대역 — 사용자별로 가진 자물쇠만 돌려준다 (SQL의 user_id 조건과 같은 효과)."""

    def __init__(self, by_user: dict[int, dict[int, MemoryRecord]]):
        self.by_user = by_user
        self.calls: list[tuple[int, list[int]]] = []

    async def get_many(self, *, user_id, record_ids):
        self.calls.append((user_id, list(record_ids)))
        owned = self.by_user.get(user_id, {})
        return {rid: owned[rid] for rid in record_ids if rid in owned}


class StubEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.1, 0.2, 0.3]
