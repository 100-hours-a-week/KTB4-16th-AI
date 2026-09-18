from app.clients.clip_client import FakeClipClient
from app.clients.embedding_client import FakeEmbeddingClient
from app.clients.llm_client import FakeLLMClient
from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.db.repositories.embedding_repository import RecordEmbeddingData
from app.schemas.embeddings import EmbeddingGenerateRequest

CLIP_DIM = 768
TEXT_DIM = 1536


class InMemoryStore:
    def __init__(self) -> None:
        self.rows: dict[int, RecordEmbeddingData] = {}

    async def upsert(self, data: RecordEmbeddingData) -> None:
        self.rows[data.record_id] = data


def build_service(store):
    from app.services.embedding_service import EmbeddingService

    return EmbeddingService(
        clip_tagger=ClipTagger(FakeClipClient(dim=CLIP_DIM)),
        embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
        mood_tagger=MusicMoodTagger(FakeLLMClient()),
        store=store,
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
