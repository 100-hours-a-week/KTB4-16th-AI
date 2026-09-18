from app.clients.clip_client import FakeClipClient
from app.clients.embedding_client import FakeEmbeddingClient
from app.clients.llm_client import FakeLLMClient
from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.db.repositories.embedding_repository import RecordEmbeddingData
from app.db.repositories.track_mood_repository import CachedMood
from app.schemas.embeddings import EmbeddingGenerateRequest

CLIP_DIM = 768
TEXT_DIM = 1536


class InMemoryStore:
    def __init__(self) -> None:
        self.rows: dict[int, RecordEmbeddingData] = {}

    async def upsert(self, data: RecordEmbeddingData) -> None:
        self.rows[data.record_id] = data


class InMemoryTrackMoods:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, CachedMood]] = {}

    async def get(self, external_track_id: str, mood_version: str) -> CachedMood | None:
        row = self.rows.get(external_track_id)
        return row[1] if row and row[0] == mood_version else None

    async def save(self, external_track_id: str, mood_version: str, mood: CachedMood) -> None:
        self.rows[external_track_id] = (mood_version, mood)


class CountingLLM(FakeLLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **kwargs) -> str:
        self.calls += 1
        return await super().complete(**kwargs)


def build_service(store, track_moods=None, llm=None):
    from app.services.embedding_service import EmbeddingService

    return EmbeddingService(
        clip_tagger=ClipTagger(FakeClipClient(dim=CLIP_DIM)),
        embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
        mood_tagger=MusicMoodTagger(llm or FakeLLMClient()),
        store=store,
        track_moods=track_moods if track_moods is not None else InMemoryTrackMoods(),
    )


async def test_generate_without_comment(record_payload):
    store = InMemoryStore()
    await build_service(store).generate(EmbeddingGenerateRequest.model_validate(record_payload))

    row = store.rows[1029]
    assert len(row.image_embedding) == CLIP_DIM
    assert len(row.music_embedding) == TEXT_DIM
    assert row.comment_embedding is None
    assert row.music_mood_text
    assert row.model_versions["moodPrompt"] == PROMPT_VERSION


async def test_blank_comment_is_treated_as_null(record_payload):
    record_payload["comment"] = "   "
    store = InMemoryStore()
    await build_service(store).generate(EmbeddingGenerateRequest.model_validate(record_payload))
    assert store.rows[1029].comment_embedding is None


async def test_generate_with_comment(record_payload):
    record_payload["comment"] = "비 오는 저녁, 창가에서"
    store = InMemoryStore()
    await build_service(store).generate(EmbeddingGenerateRequest.model_validate(record_payload))
    assert len(store.rows[1029].comment_embedding) == TEXT_DIM


async def test_same_input_gives_same_vector(record_payload):
    store_a, store_b = InMemoryStore(), InMemoryStore()
    req = EmbeddingGenerateRequest.model_validate(record_payload)
    await build_service(store_a).generate(req)
    await build_service(store_b).generate(req)
    assert store_a.rows[1029].image_embedding == store_b.rows[1029].image_embedding


async def test_all_dummy_records_are_valid_requests():
    import json
    from pathlib import Path

    records = json.loads(Path("fixtures/dummy_records.json").read_text(encoding="utf-8"))
    assert records
    for record in records:
        EmbeddingGenerateRequest.model_validate(record)


async def test_same_track_calls_llm_once(record_payload):
    store, moods, llm = InMemoryStore(), InMemoryTrackMoods(), CountingLLM()
    service = build_service(store, moods, llm)
    for record_id in (1, 2, 3):
        payload = {**record_payload, "recordId": record_id}
        await service.generate(EmbeddingGenerateRequest.model_validate(payload))
    assert llm.calls == 1
    assert store.rows[1].music_embedding == store.rows[3].music_embedding
    assert store.rows[2].external_track_id == "6rqhFgbbKwnb9MLmUQDhG6"


async def test_mood_is_rebuilt_when_version_changes(record_payload):
    moods, llm = InMemoryTrackMoods(), CountingLLM()
    moods.rows["6rqhFgbbKwnb9MLmUQDhG6"] = ("old-version", CachedMood("옛 묘사", [0.0] * TEXT_DIM))
    await build_service(InMemoryStore(), moods, llm).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )
    assert llm.calls == 1


def test_track_id_uri_prefix_is_stripped(record_payload):
    record_payload["track"]["externalTrackId"] = "spotify:track:abc123"
    req = EmbeddingGenerateRequest.model_validate(record_payload)
    assert req.track.external_track_id == "abc123"
