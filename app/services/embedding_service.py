"""임베딩 인덱싱 처리 순서: 사진·음악·코멘트 세 갈래를 병렬로 만들고 한 행으로 저장."""

import asyncio

from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.db.repositories.embedding_repository import EmbeddingStore, RecordEmbeddingData
from app.db.repositories.track_mood_repository import CachedMood, TrackMoodStore
from app.schemas.embeddings import EmbeddingGenerateRequest, EmbeddingTrack


class EmbeddingService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        embedder: Embedder,
        mood_tagger: MusicMoodTagger,
        store: EmbeddingStore,
        track_moods: TrackMoodStore,
    ):
        self._clip = clip_tagger
        self._embedder = embedder
        self._mood = mood_tagger
        self._store = store
        self._track_moods = track_moods

    @property
    def mood_version(self) -> str:
        # 이 셋 중 하나라도 바뀌면 캐시된 무드는 다시 만든다
        return f"{self._mood.model}|{PROMPT_VERSION}|{self._embedder.model}"

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
                    "moodLlm": self._mood.model,
                    "moodPrompt": PROMPT_VERSION,
                },
            )
        )

    async def _embed_music(self, track: EmbeddingTrack) -> tuple[str, list[float]]:
        """같은 곡은 무드를 한 번만 만든다 — 스포티파이 곡 ID로 캐시를 먼저 찾는다."""
        cached = await self._track_moods.get(track.external_track_id, self.mood_version)
        if cached is not None:
            return cached.mood_text, cached.mood_embedding

        mood_text = await self._mood.describe(track.title, track.artist_name)
        mood_vec = await self._embedder.embed(mood_text)
        await self._track_moods.save(
            track.external_track_id,
            self.mood_version,
            CachedMood(mood_text=mood_text, mood_embedding=mood_vec),
        )
        return mood_text, mood_vec

    async def _embed_comment(self, comment: str | None) -> list[float] | None:
        # null·공백 코멘트는 정상 상태 — 벡터 없이 저장
        if comment is None or not comment.strip():
            return None
        return await self._embedder.embed(comment)
