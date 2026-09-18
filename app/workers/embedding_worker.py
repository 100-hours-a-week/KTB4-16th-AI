"""임베딩 인덱싱 작업 실행 단위."""

from typing import Any

from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.music_mood_tagger import MusicMoodTagger
from app.db.postgres import get_sessionmaker
from app.db.repositories.embedding_repository import EmbeddingRepository
from app.db.repositories.track_mood_repository import TrackMoodRepository
from app.dependencies import get_clients
from app.schemas.embeddings import EmbeddingGenerateRequest
from app.services.embedding_service import EmbeddingService

JOB_TYPE = "embedding"


async def handle(payload: dict[str, Any]) -> None:
    req = EmbeddingGenerateRequest.model_validate(payload)
    clients = get_clients()
    async with get_sessionmaker()() as session:
        service = EmbeddingService(
            clip_tagger=ClipTagger(clients.clip),
            embedder=Embedder(clients.embedding),
            mood_tagger=MusicMoodTagger(clients.llm_knowledge),
            store=EmbeddingRepository(session),
            track_moods=TrackMoodRepository(session),
        )
        await service.generate(req)
