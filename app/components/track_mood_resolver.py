"""곡 단위 무드 조회/생성 — 기능4(임베딩)와 재랭킹 백필 워커가 공유.

"같은 곡이면 LLM을 한 번만 부른다"는 규칙을 한 곳에만 둔다. 두 군데서
각자 구현하면 mood_version 계산식이 미묘하게 달라질 때 캐시가 서로
안 맞게 된다.
"""

from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.db.repositories.track_mood_repository import CachedMood, TrackMoodStore


class TrackMoodResolver:
    def __init__(self, mood_tagger: MusicMoodTagger, embedder: Embedder, store: TrackMoodStore):
        self._mood = mood_tagger
        self._embedder = embedder
        self._store = store

    @property
    def mood_version(self) -> str:
        # 이 셋 중 하나라도 바뀌면 캐시된 무드는 다시 만든다
        return f"{self._mood.model}|{PROMPT_VERSION}|{self._embedder.model}"

    async def resolve(self, *, title: str, artist_name: str, external_track_id: str) -> CachedMood:
        cached = await self._store.get(external_track_id, self.mood_version)
        if cached is not None:
            return cached

        description = await self._mood.describe(title, artist_name)
        mood_vec = await self._embedder.embed(description.text)
        mood = CachedMood(
            mood_text=description.text,
            mood_embedding=mood_vec,
            genre=description.genre,
            moods=description.moods,
        )
        await self._store.save(external_track_id, self.mood_version, mood)
        return mood
