"""임베딩 인덱싱 처리 순서: 사진·음악·코멘트 세 갈래를 병렬로 만들고 한 행으로 저장.

저장할 때 사진 벡터로 사진을 한 번 분류해 태그(장소 + 다축)도 같이 저장한다. RECAP 같은 곳이
저장된 태그를 세기만 하면 되고, 태그 목록을 바꿨을 때도 저장된 벡터로 다시 분류할 수 있다.
"""

import asyncio
import logging

from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.track_mood_resolver import TrackMoodResolver
from app.db.repositories.embedding_repository import EmbeddingStore, RecordEmbeddingData
from app.exceptions import ClipError
from app.schemas.embeddings import EmbeddingGenerateRequest, EmbeddingTrack

logger = logging.getLogger("muro.embedding")

# photo_tags에서 장소(mood_tags.yaml) 태그의 키. 다축 태그는 축 이름
# (사진 종류·대상·분위기·표정·감정)을 키로 쓴다
SCENE_KEY = "장소"


class EmbeddingService:
    def __init__(
        self,
        clip_tagger: ClipTagger,
        embedder: Embedder,
        track_moods: TrackMoodResolver,
        store: EmbeddingStore,
        tag_version: str | None = None,
    ):
        self._clip = clip_tagger
        self._embedder = embedder
        self._track_moods = track_moods
        self._store = store
        self._tag_version = tag_version

    async def generate(self, req: EmbeddingGenerateRequest) -> None:
        image_vec, (mood_text, music_vec), comment_vec = await asyncio.gather(
            self._clip.encode_image(req.photo_url),
            self._embed_music(req.track),
            self._embed_comment(req.comment),
        )
        photo_tags = await self._classify_photo(image_vec)
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
                    **({"tagVersion": self._tag_version} if self._tag_version else {}),
                },
                photo_tags=photo_tags,
            )
        )

    async def _classify_photo(self, image_vector: list[float]) -> dict[str, str] | None:
        """저장된 벡터로 사진을 분류한다 (CLIP 이미지 호출 없음). 실패하면 None.

        태그는 부가 정보라서, 태그 문장 인코딩이 실패(ClipError)해도 임베딩 저장은 계속한다.
        None으로 저장된 행은 태그 목록이 준비된 뒤 저장된 벡터로 다시 분류하면 된다.
        """
        try:
            scene = await self._clip.tag_from_vector(image_vector, top_k=1)
            axes = await self._clip.classify_axes(image_vector)
        except ClipError:
            logger.warning("사진 태그를 만들지 못해 태그 없이 저장해요.", exc_info=True)
            return None
        return {**({SCENE_KEY: scene[0]} if scene else {}), **axes}

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
