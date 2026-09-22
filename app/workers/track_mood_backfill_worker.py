"""재랭킹에서 캐시에 없던 곡의 '진짜' 트랙무드를 뒤늦게 만들어 채워두는 워커.

기능3(사진 추천)의 동기 경로는 지연 예산 때문에 이 곡들을 원문 임베딩으로만
빠르게 처리하고 넘어간다. 여기서 LLM으로 정식 무드를 만들어 캐시에 저장해두면,
다음에 같은 곡이 후보로 나올 때는 캐시가 적중해 품질이 올라간다.
"""

from typing import Any

from app.components.embedder import Embedder
from app.components.music_mood_tagger import MusicMoodTagger
from app.components.reranker import BACKFILL_JOB_TYPE
from app.components.track_mood_resolver import TrackMoodResolver
from app.db.postgres import get_sessionmaker
from app.db.repositories.track_mood_repository import TrackMoodRepository
from app.dependencies import get_clients

JOB_TYPE = BACKFILL_JOB_TYPE


async def handle(payload: dict[str, Any]) -> None:
    clients = get_clients()
    async with get_sessionmaker()() as session:
        resolver = TrackMoodResolver(
            mood_tagger=MusicMoodTagger(clients.llm_knowledge),
            embedder=Embedder(clients.embedding),
            store=TrackMoodRepository(session),
        )
        await resolver.resolve(
            title=payload["title"],
            artist_name=payload["artistName"],
            external_track_id=payload["externalTrackId"],
        )
