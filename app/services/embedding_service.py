"""임베딩 인덱싱 처리 순서: 사진·음악·코멘트 세 갈래를 병렬로 만들고 한 행으로 저장."""

import asyncio

from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.track_mood_resolver import TrackMoodResolver
from app.db.repositories.embedding_repository import EmbeddingStore, RecordEmbeddingData
from app.schemas.embeddings import EmbeddingGenerateRequest, EmbeddingTrack


class EmbeddingService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        embedder: Embedder,
        track_moods: TrackMoodResolver,
        store: EmbeddingStore,
    ):
        self._clip = clip_tagger
        self._embedder = embedder
        self._track_moods = track_moods
        self._store = store

    async def generate(self, req: EmbeddingGenerateRequest) -> None:
        image_vec, (mood_text, music_vec), comment_vec = await asyncio.gather(
            self._clip.encode_image(req.photo_url),
            self._embed_music(req.track),
            self._embed_comment(req.comment),
        )
        await self._store.upsert(
            RecordEmbeddingData(
                record_id=req.record_id,
                user_id=req.user_id,
                external_track_id=req.track.external_track_id,
                image_embedding=image_vec,
                music_embedding=music_vec,
                comment_embedding=comment_vec,
                music_mood_text=mood_text,
                model_versions={
                    "image": self._clip.model,
                    "text": self._embedder.model,
                    "moodVersion": self._track_moods.mood_version,
                },
            )
        )

    async def _embed_music(self, track: EmbeddingTrack) -> tuple[str, list[float]]:
        """같은 곡은 무드를 한 번만 만든다 — 스포티파이 곡 ID로 캐시를 먼저 찾는다."""
        mood = await self._track_moods.resolve(
            title=track.title,
            artist_name=track.artist_name,
            external_track_id=track.external_track_id,
        )
        return mood.mood_text, mood.mood_embedding

    async def _embed_comment(self, comment: str | None) -> list[float] | None:
        # null·공백 코멘트는 정상 상태 — 벡터 없이 저장
        if comment is None or not comment.strip():
            return None
        return await self._embedder.embed(comment)
