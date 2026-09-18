"""record_embeddings 접근."""

from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RecordEmbedding


@dataclass(frozen=True)
class RecordEmbeddingData:
    record_id: int
    user_id: int
    image_embedding: list[float]
    music_embedding: list[float]
    comment_embedding: list[float] | None
    music_mood_text: str
    model_versions: dict[str, Any]


class EmbeddingStore(Protocol):
    async def upsert(self, data: RecordEmbeddingData) -> None: ...


class EmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert(self, data: RecordEmbeddingData) -> None:
        """같은 recordId가 다시 와도 1행만 유지한다 (멱등)."""
        values = {
            "record_id": data.record_id,
            "user_id": data.user_id,
            "image_embedding": data.image_embedding,
            "music_embedding": data.music_embedding,
            "comment_embedding": data.comment_embedding,
            "music_mood_text": data.music_mood_text,
            "model_versions": data.model_versions,
        }
        stmt = insert(RecordEmbedding).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[RecordEmbedding.record_id],
            set_={
                **{k: stmt.excluded[k] for k in values if k != "record_id"},
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()
