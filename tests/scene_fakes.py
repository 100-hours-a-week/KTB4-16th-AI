# ruff: noqa: E501
"""장면 문장 방식(기능1·3) 시험용 가짜들 — 곡풀·장면·임베딩·Spotify."""

import random

from app.components.scene_curator import PhotoReading, Scene
from app.components.song_pool import PoolSong, SongPool

DIM = 3


class StubEmbedder:
    """모든 문장을 같은 벡터로 바꾼다 — 곡 순위는 곡 벡터 차이로만 갈린다."""

    model = "stub-embed"

    def __init__(self, error: Exception | None = None, vector=(1.0, 0.0, 0.0)):
        self._error = error
        self._vector = list(vector)
        self.texts: list[str] = []
        self.batches: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        if self._error:
            raise self._error
        return list(self._vector)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self._error:
            raise self._error
        return [list(self._vector) for _ in texts]


class StubSceneCurator:
    def __init__(
        self,
        scenes: list[Scene] | None = None,
        error: Exception | None = None,
        summary: str = "노을 지는 바닷가",
    ):
        self.scenes = scenes if scenes is not None else SCENES
        self._error = error
        self.summary = summary
        self.situations: list[str] = []

    async def for_context(self, situation: str) -> list[Scene]:
        self.situations.append(situation)
        if self._error:
            raise self._error
        return self.scenes

    async def for_photo(self, image_url: str) -> PhotoReading:
        self.situations.append(image_url)
        if self._error:
            raise self._error
        return PhotoReading(self.scenes, self.summary)


SCENES = [
    Scene("퇴근길 버스 창가에서 비를 보는 직장인", "비 오는 저녁을 닮은 곡이다. 가. 나.", "비가 와 / 창가에서 / 너를 떠올려 / 괜찮아"),
    Scene("카페에서 책을 읽는 대학생", "잔잔한 오후를 닮은 곡이다. 다. 라.", "책을 덮고 / 커피를 마셔 / 천천히 / 웃어"),
    Scene("집에서 혼자 쉬는 사람", "조용한 밤을 닮은 곡이다. 마. 바.", "불을 끄고 / 누워서 / 눈을 감아 / 잠들어"),
]  # fmt: skip  # noqa: E501


def song(artist, title, genre="발라드", vector=(1.0, 0.0, 0.0), desc=None, hub=0.0) -> PoolSong:
    v = tuple(vector)
    return PoolSong(
        artist=artist,
        title=title,
        genres=(genre,),
        moods=(),
        embedding=v,
        lyric_embedding=v,
        description_embedding=tuple(desc) if desc is not None else v,
        hub=hub,
    )


def pool_of(songs: list[PoolSong], seed: int = 0) -> SongPool:
    return SongPool(songs, rng=random.Random(seed))


def big_pool() -> tuple[SongPool, list[PoolSong]]:
    """발라드 55곡 + 댄스 55곡 + 록 20곡 — 취향 거르기의 최소 후보 수(50)를 넘긴다.

    질문 벡터(1,0,0)와 가까운 순서는 발라드 > 댄스 > 록이 되게 벡터를 정했다.
    """
    songs = []
    for genre, base in (("발라드", 0.0), ("댄스", 0.4), ("록/메탈", 0.8)):
        count = 20 if genre == "록/메탈" else 55
        for i in range(count):
            y = base + i * 0.001
            songs.append(song(f"{genre}가수{i}", f"{genre}곡{i}", genre, (1.0, y, 0.0)))
    return pool_of(songs), songs


class StubImages:
    """사진 내려받기 — 항상 같은 data URL을 준다."""

    def __init__(self, error: Exception | None = None):
        self._error = error
        self.urls: list[str] = []

    async def fetch_data_url(self, url: str) -> str:
        self.urls.append(url)
        if self._error:
            raise self._error
        return "data:image/jpeg;base64,AAAA"


class StubSpotify:
    """곡 풀 곡 모두 찾아 준다. 찾을 수 없는 곡·장애는 만들어 준 dict로 정한다."""

    def __init__(self, songs: list[PoolSong] | None = None, error: Exception | None = None):
        self._error = error
        self._by_query = {}
        for i, s in enumerate(songs or []):
            self._by_query[f"track:{s.title} artist:{s.artist}"] = [
                track_json(f"t_{i}", s.title, s.artist)
            ]
        self.queries: list[str] = []

    async def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        self.queries.append(query)
        if self._error:
            raise self._error
        return self._by_query.get(query, [])

    async def get_track(self, track_id: str) -> dict:
        if self._error:
            raise self._error
        return track_json(track_id, "참고곡", "참고가수")


def track_json(track_id: str, title: str, artist: str) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": [{"name": artist}],
        "uri": f"spotify:track:{track_id}",
        "album": {"images": [{"url": f"https://i.scdn.co/image/{track_id}"}]},
    }
