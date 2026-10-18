from dataclasses import replace
from pathlib import Path

from app.clients.clip_client import FakeClipClient
from app.clients.embedding_client import FakeEmbeddingClient
from app.clients.llm_client import FakeLLMClient
from app.components.axis_vectors import AxisVectors
from app.components.category_vectors import CategoryVectors
from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.components.track_mood_resolver import TrackMoodResolver
from app.db.repositories.embedding_repository import RecordEmbeddingData
from app.db.repositories.track_mood_repository import CachedMood
from app.exceptions import ClipError
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

    async def get_any(self, external_track_id: str) -> CachedMood | None:
        row = self.rows.get(external_track_id)
        return row[1] if row else None

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
        clip_tagger=ClipTagger(
            FakeClipClient(dim=CLIP_DIM),
            CategoryVectors(FakeClipClient(dim=CLIP_DIM)),
            AxisVectors(FakeClipClient(dim=CLIP_DIM)),
        ),
        embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
        track_moods=TrackMoodResolver(
            mood_tagger=MusicMoodTagger(llm or FakeLLMClient()),
            embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
            store=track_moods if track_moods is not None else InMemoryTrackMoods(),
        ),
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
    assert PROMPT_VERSION in row.model_versions["moodVersion"]


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
    moods.rows["6rqhFgbbKwnb9MLmUQDhG6"] = (
        "old-version",
        CachedMood("옛 묘사", [0.0] * TEXT_DIM, genre="발라드", moods=("잔잔한",)),
    )
    await build_service(InMemoryStore(), moods, llm).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )
    assert llm.calls == 1


def test_track_id_uri_prefix_is_stripped(record_payload):
    record_payload["track"]["externalTrackId"] = "spotify:track:abc123"
    req = EmbeddingGenerateRequest.model_validate(record_payload)
    assert req.track.external_track_id == "abc123"


# --- 사진 태그 저장 ---------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SCENE_YAML = FIXTURES / "mood_tags_sample.yaml"
AXIS_YAML = FIXTURES / "axis_tags_sample.yaml"

# 벡터 6칸: [진짜사진, 그림, 음식, 풍경, 웃음, 슬픔] (장소 라벨은 3번째 칸이 노을, 4번째가 밤거리)
TEXT_VECTORS = {
    "sunset": [0, 0, 1.0, 0, 0, 0],
    "night street": [0, 0, 0, 1.0, 0, 0],
    "real photo": [1.0, 0, 0, 0, 0, 0],
    "camera photo": [1.0, 0, 0, 0, 0, 0],
    "cartoon": [0, 1.0, 0, 0, 0, 0],
    "food": [0, 0, 1.0, 0, 0, 0],
    "landscape": [0, 0, 0, 1.0, 0, 0],
    "laugh": [0, 0, 0, 0, 1.0, 0],
    "cry": [0, 0, 0, 0, 0, 1.0],
}


class TaggingClip(FakeClipClient):
    """사진 벡터를 [진짜 사진 + 음식/노을 방향]으로 고정한다."""

    def __init__(self, text_error: Exception | None = None):
        super().__init__(dim=CLIP_DIM, model="tagging-clip")
        self._text_error = text_error
        self.image_calls = 0

    async def encode_image(self, image_url: str) -> list[float]:
        self.image_calls += 1
        return [1.0, 0, 1.0, 0, 0, 0] + [0.0] * (CLIP_DIM - 6)

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        if self._text_error:
            raise self._text_error
        return [TEXT_VECTORS[t] + [0.0] * (CLIP_DIM - 6) for t in texts]


def build_tagging_service(store, clip, tag_version="tv-1"):
    from app.services.embedding_service import EmbeddingService

    return EmbeddingService(
        clip_tagger=ClipTagger(
            clip,
            CategoryVectors(clip, path=SCENE_YAML),
            AxisVectors(clip, path=AXIS_YAML),
        ),
        embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
        track_moods=TrackMoodResolver(
            mood_tagger=MusicMoodTagger(FakeLLMClient()),
            embedder=Embedder(FakeEmbeddingClient(dim=TEXT_DIM)),
            store=InMemoryTrackMoods(),
        ),
        store=store,
        tag_version=tag_version,
    )


async def test_photo_tags_are_saved_with_scene_and_axes(record_payload):
    store, clip = InMemoryStore(), TaggingClip()

    await build_tagging_service(store, clip).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )

    row = store.rows[1029]
    assert row.photo_tags == {"장소": "노을", "사진 종류": "진짜 사진", "대상": "음식"}
    assert clip.image_calls == 1  # 태그를 위해 사진 CLIP을 다시 부르지 않는다


async def test_tag_version_is_saved_next_to_the_other_versions(record_payload):
    store = InMemoryStore()

    await build_tagging_service(store, TaggingClip(), tag_version="tv-abc").generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )

    versions = store.rows[1029].model_versions
    assert versions["tagVersion"] == "tv-abc"
    assert versions["image"] == "tagging-clip"  # 기존 값은 그대로


async def test_embedding_is_still_saved_when_tags_cannot_be_made(record_payload):
    """태그 문장 인코딩이 실패해도(ClipError) 임베딩 저장은 성공하고 태그만 비운다(NULL)."""

    class TextFails(TaggingClip):
        async def encode_text(self, texts: list[str]) -> list[list[float]]:
            raise ClipError("태그 문장 인코딩 실패")

    store = InMemoryStore()

    await build_tagging_service(store, TextFails()).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )

    row = store.rows[1029]
    assert row.photo_tags is None
    assert len(row.image_embedding) == CLIP_DIM
    assert len(row.music_embedding) == TEXT_DIM


async def test_photo_tags_is_empty_dict_when_nothing_is_confident(record_payload):
    """분류는 했지만 붙일 태그가 없는 것({})과 분류하지 못한 것(None)을 구분한다."""

    class BlankPhoto(TaggingClip):
        async def encode_image(self, image_url: str) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * (
                CLIP_DIM - 6
            )  # 어떤 태그와도 거의 직교

    store = InMemoryStore()

    await build_tagging_service(store, BlankPhoto()).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )

    # 사진 종류가 불확실해서 다축 태그는 없고, 장소 태그도 임계치를 넘는 것이 없다
    assert store.rows[1029].photo_tags == {}


async def test_service_without_tag_version_leaves_model_versions_as_before(record_payload):
    store = InMemoryStore()
    await build_service(store).generate(EmbeddingGenerateRequest.model_validate(record_payload))

    assert "tagVersion" not in store.rows[1029].model_versions


async def test_new_track_mood_stores_title_and_artist(record_payload):
    moods = InMemoryTrackMoods()
    await build_service(InMemoryStore(), moods, CountingLLM()).generate(
        EmbeddingGenerateRequest.model_validate(record_payload)
    )
    saved = moods.rows["6rqhFgbbKwnb9MLmUQDhG6"][1]
    assert saved.title == record_payload["track"]["title"]
    assert saved.artist_name == record_payload["track"]["artistName"]


async def test_cached_track_without_names_gets_them_filled_without_calling_llm(record_payload):
    moods, llm = InMemoryTrackMoods(), CountingLLM()
    service = build_service(InMemoryStore(), moods, llm)
    await service.generate(EmbeddingGenerateRequest.model_validate(record_payload))
    key = "6rqhFgbbKwnb9MLmUQDhG6"
    version, mood = moods.rows[key]
    moods.rows[key] = (version, replace(mood, title=None, artist_name=None))  # 칸이 생기기 전 행

    await service.generate(EmbeddingGenerateRequest.model_validate(record_payload))

    assert llm.calls == 1
    assert moods.rows[key][1].title == record_payload["track"]["title"]
