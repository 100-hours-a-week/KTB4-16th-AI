"""기능5 기억 검색 테스트용 가짜 구현 — 벡터 검색, 백엔드 상세 조회, 임베더."""

from datetime import datetime

from app.components.vector_search import SimilarRecord
from app.db.repositories.memory_record_repository import ConditionRow, MemoryRecord


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


def make_row(record_id: int, **overrides) -> ConditionRow:
    values = {
        "record_id": record_id,
        "created_at": datetime(2026, 10, 3, 18, 30),
        "weather_condition": "CLEAR",
        "temperature": 18.5,
        "mood_score": 10,
        "dong_name": "성수동1가",
        "dong_code": "1120011500",
        "latitude": 37.5440,
        "longitude": 127.0370,
    }
    values.update(overrides)
    return ConditionRow(**values)


class FakeVectorSearch:
    """미리 정한 결과를 돌려주고, 어떤 사용자·개수로 불렀는지 기록한다."""

    def __init__(
        self,
        comment=None,
        music=None,
        *,
        moods: dict[int, tuple[str, ...]] | None = None,
        comment_similarity: dict[int, float] | None = None,
        music_similarity: dict[int, float] | None = None,
    ):
        self.comment = comment or []
        self.music = music or []
        self.moods = moods or {}
        # similarities_for가 돌려줄 값 (record_id → 유사도). 없는 기록은 결과에서 빠진다.
        self.comment_similarity = comment_similarity or {}
        self.music_similarity = music_similarity or {}
        self.calls: list[tuple[str, int, int]] = []
        self.similarity_calls: list[tuple[int, list[int]]] = []
        self.mood_calls: list[int] = []

    async def similar_by_comment(self, user_id, query_vector, limit):
        self.calls.append(("comment", user_id, limit))
        return self.comment

    async def similar_by_music(self, user_id, query_vector, limit):
        self.calls.append(("music", user_id, limit))
        return self.music

    async def similarities_for(self, user_id, record_ids, query_vector):
        self.similarity_calls.append((user_id, list(record_ids)))
        return (
            [
                hit(rid, self.comment_similarity[rid])
                for rid in record_ids
                if rid in self.comment_similarity
            ],
            [
                hit(rid, self.music_similarity[rid])
                for rid in record_ids
                if rid in self.music_similarity
            ],
        )

    async def moods_by_record(self, user_id):
        self.mood_calls.append(user_id)
        return dict(self.moods)


class FakeMemoryRecords:
    """백엔드 MySQL 대역 — 사용자별로 가진 자물쇠만 돌려준다 (SQL의 user_id 조건과 같은 효과)."""

    def __init__(
        self,
        by_user: dict[int, dict[int, MemoryRecord]],
        rows_by_user: dict[int, list[ConditionRow]] | None = None,
    ):
        self.by_user = by_user
        self.rows_by_user = rows_by_user or {}
        self.calls: list[tuple[int, list[int]]] = []
        self.row_calls: list[tuple[int, int]] = []

    async def get_many(self, *, user_id, record_ids):
        self.calls.append((user_id, list(record_ids)))
        owned = self.by_user.get(user_id, {})
        return {rid: owned[rid] for rid in record_ids if rid in owned}

    async def list_condition_rows(self, *, user_id, limit):
        self.row_calls.append((user_id, limit))
        return list(self.rows_by_user.get(user_id, []))[:limit]


class StubEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.1, 0.2, 0.3]
