"""곡 풀 — 미리 장르·무드를 붙여 둔 곡 목록에서 사진 상황에 맞는 곡을 고른다 (기능3).

사진 추천 5곡을 전부 LLM이 고르면 매번 비슷한 유명곡 위주가 되고, LLM이 지어낸 곡은
Spotify 확인에서 탈락한다. 그래서 5곡 중 일부를 이 곡 풀에서 채운다. 곡 풀 곡은 실제
노래방 인기 차트(1년치)에 있던 곡이라 존재가 확실하고, 장르는 차트 분류, 무드·설명은
곡을 아는 경우에만 붙였다 (모르는 곡은 뺐다). 만드는 법은 scripts/build_song_pool.py.

고르는 방식은 reranker와 같은 "정해진 태그 겹침"이다 — 자유 문장 임베딩은 신나는지
잔잔한지를 잘 구분 못 했다(reranker.py 참고). 같은 상황이면 매번 같은 곡이 나오지
않게 점수 상위 몇 곡 중에서 무작위로 고른다.

곡 풀 파일은 공개 레포에 두지 않고 비공개 HF 데이터 저장소(dionypark/mulo-song-pool)에
두며, Docker 빌드 때 받아 온다(Dockerfile, 설정 song_pool_path).

곡 풀 파일에는 Spotify 곡 ID가 없다 — 뽑힌 곡은 LLM 곡과 똑같이 TrackResolver로 찾고,
찾은 결과는 track_lookups에 캐시된다.
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.components.song_curator import SongCandidate

logger = logging.getLogger("muro.song_pool")

GENRE_WEIGHT = 0.5
MOOD_WEIGHT = 0.5
# 장르가 맞거나 무드가 상황 무드의 절반 이상 겹쳐야 후보 — 억지로 끼워 넣지 않는다
MIN_SCORE = 0.5
# 점수 상위 몇 곡 중에서 무작위로 고를지
TOP_K = 15


@dataclass(frozen=True)
class PoolSong:
    artist: str
    title: str
    genres: tuple[str, ...]  # 첫 번째가 주 장르
    moods: tuple[str, ...]
    popularity: int = 0  # 0~100, 차트에 오른 개월 수 기준
    # 설명 문장의 KURE 벡터(길이 1) — 없으면 태그 점수로만 고른다 (pool-v2 이하 파일)
    embedding: tuple[float, ...] | None = None
    # 장면 문장으로 고르는 방식(rank_scenes)에서 쓰는 두 벡터. 가사 벡터와 곡 설명 벡터 모두 KURE,
    # 파일에 없으면 None. embedding은 이 중 가사 → 설명 순으로 먼저 있는 값이다 (예전 방식용)
    lyric_embedding: "np.ndarray | None" = None
    description_embedding: "np.ndarray | None" = None
    song_id: str = ""
    # 허브 값 — 여러 상황의 장면 문장(곡풀 만들 때 쓴 300개)과 비교한 이 곡의 평균 점수.
    # 어느 상황에나 점수가 높게 나오는 곡은 이 값이 커서, 점수에서 일부를 빼 준다 (rank_scenes)
    hub: float = 0.0

    def to_candidate(self) -> SongCandidate:
        return SongCandidate(
            artist=self.artist, title=self.title, genre=self.genres[0], moods=self.moods
        )


@dataclass(frozen=True)
class ScoredSong:
    """장면 문장으로 매긴 곡 점수. scene_index는 이 곡이 가장 잘 맞은 장면(0부터)."""

    song: PoolSong
    score: float
    scene_index: int


class SongPool:
    def __init__(self, songs: list[PoolSong], rng: random.Random | None = None):
        self._songs = songs
        self._rng = rng or random.Random()
        self._matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def _vector_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(가사 행렬, 가사 있음, 설명 행렬, 설명 있음) — 처음 한 번만 만든다."""
        if self._matrices is None:
            dim = next(
                (
                    len(v)
                    for s in self._songs
                    for v in (s.lyric_embedding, s.description_embedding)
                    if v is not None
                ),
                0,
            )

            def build(attr: str) -> tuple[np.ndarray, np.ndarray]:
                matrix = np.zeros((len(self._songs), dim), dtype=np.float32)
                has = np.zeros(len(self._songs), dtype=bool)
                for i, song in enumerate(self._songs):
                    vector = getattr(song, attr)
                    if vector is not None and len(vector) == dim:
                        matrix[i] = vector
                        has[i] = True
                return matrix, has

            lyric, has_lyric = build("lyric_embedding")
            desc, has_desc = build("description_embedding")
            self._matrices = (lyric, has_lyric, desc, has_desc)
        return self._matrices

    def rank_scenes(
        self,
        lyric_queries: list[list[float]],
        description_queries: list[list[float]],
        exclude: list[SongCandidate] | None = None,
        lyric_weight: float = 0.5,
        hub_weight: float = 0.0,
    ) -> list[ScoredSong]:
        """장면마다 (가사 문체 문장 ↔ 가사 벡터, 설명 문체 문장 ↔ 설명 벡터) 점수를 각각 z점수로
        바꿔 lyric_weight : (1 - lyric_weight)로 합치고, 장면 3개의 평균으로 곡 점수를 매긴다.

        z점수는 장면마다 곡 전체에서 따로 낸다 — 벡터 종류마다 점수 분포가 달라서 그냥 더하면 한쪽이
        묻힌다. 곡 점수에서 hub_weight × (그 곡의 허브 값)을 뺀다 — 어느 상황에나 높게 나오는 곡이
        매번 1등이 되는 것을 줄인다 (50개 상황 실측: 0.5에서 점수 그대로, 곡 종류 86→140).
        한 벡터만 있는 곡은 그 벡터 점수를 양쪽에 쓴다. 질문 벡터 길이가 곡 벡터와 다르면
        (임베딩 모델이 바뀐 경우) 빈 목록 — 서비스가 폴백한다. 점수 높은 순으로 돌려준다.
        """
        lyric, has_lyric, desc, has_desc = self._vector_matrices()
        if not len(self._songs) or lyric.shape[1] == 0:
            return []
        queries_l = np.asarray(lyric_queries, dtype=np.float32)
        queries_d = np.asarray(description_queries, dtype=np.float32)
        if (
            queries_l.ndim != 2
            or queries_d.ndim != 2
            or not len(queries_l)
            or len(queries_l) != len(queries_d)
            or queries_l.shape[1] != lyric.shape[1]
            or queries_d.shape[1] != lyric.shape[1]
        ):
            return []

        def z(raw: np.ndarray, has: np.ndarray) -> np.ndarray:
            out = np.zeros_like(raw)
            if has.any():
                part = raw[has]
                out[has] = (part - part.mean(axis=0)) / (part.std(axis=0) + 1e-9)
            return out

        z_l = z(lyric @ queries_l.T, has_lyric)
        z_d = z(desc @ queries_d.T, has_desc)
        z_l = np.where(has_lyric[:, None], z_l, z_d)
        z_d = np.where(has_desc[:, None], z_d, z_l)
        per_scene = lyric_weight * z_l + (1 - lyric_weight) * z_d
        score = per_scene.mean(axis=1)
        if hub_weight:
            score = score - hub_weight * np.array(
                [song.hub for song in self._songs], dtype=np.float32
            )
        best = per_scene.argmax(axis=1)

        excluded = {_song_key(s.artist, s.title) for s in exclude or []}
        usable = has_lyric | has_desc
        out_list = [
            ScoredSong(song, float(score[i]), int(best[i]))
            for i, song in enumerate(self._songs)
            if usable[i] and _song_key(song.artist, song.title) not in excluded
        ]
        out_list.sort(key=lambda r: r.score, reverse=True)
        return out_list

    def __len__(self) -> int:
        return len(self._songs)

    def pick(
        self,
        situation_genres: tuple[str, ...],
        situation_moods: tuple[str, ...],
        count: int,
        exclude: list[SongCandidate] | None = None,
        situation_vector: list[float] | None = None,
    ) -> list[SongCandidate]:
        """상황에 맞는 곡 count개. 맞는 곡이 모자라면 있는 만큼만 (빈 목록일 수 있다).

        1단계는 태그(장르·무드)로 후보를 거르고, situation_vector가 있으면 2단계로 그 후보들을
        분위기 문장 벡터와 가까운 순서로 다시 줄 세운다. 벡터가 없거나 곡에 벡터가 없으면
        (모델이 다르면 차원이 달라 비교가 안 된다) 예전처럼 태그 점수 순이다.
        """
        if count <= 0:
            return []
        excluded = {_song_key(s.artist, s.title) for s in exclude or []}
        scored = [
            (score, song)
            for song in self._songs
            if _song_key(song.artist, song.title) not in excluded
            and (score := _score(situation_genres, situation_moods, song)) >= MIN_SCORE
        ]
        if situation_vector is not None and any(
            cosine_rank_usable(situation_vector, song.embedding) for _, song in scored
        ):
            scored = [
                (cosine_similarity(situation_vector, song.embedding), song) for _, song in scored
            ]
        scored.sort(key=lambda pair: (pair[0], pair[1].popularity), reverse=True)
        top = [song for _, song in scored[:TOP_K]]
        self._rng.shuffle(top)

        picked: list[PoolSong] = []
        artists: set[str] = set()
        for song in top:
            artist = _norm(song.artist)
            if artist in artists:
                continue
            artists.add(artist)
            picked.append(song)
            if len(picked) == count:
                break
        return [song.to_candidate() for song in picked]

    def ranked(
        self,
        situation_genres: tuple[str, ...],
        situation_moods: tuple[str, ...],
        exclude: list[SongCandidate] | None = None,
        situation_vector: list[float] | None = None,
    ) -> list[tuple[PoolSong, float, float | None]]:
        """태그 기준(MIN_SCORE)을 넘는 모든 후보를 (곡, 태그 점수, 벡터 유사도)로 돌려준다.

        pick()과 달리 몇 곡을 고를지·무작위·같은 가수 걸러내기는 여기서 하지 않는다 — 저장된 곡과
        같은 점수로 겨루게 하려고(db_picker) 점수만 매겨서 넘긴다. 벡터가 없거나 차원이 다르면
        유사도는 None이다.
        """
        excluded = {_song_key(s.artist, s.title) for s in exclude or []}
        usable = situation_vector is not None
        out: list[tuple[PoolSong, float, float | None]] = []
        for song in self._songs:
            if _song_key(song.artist, song.title) in excluded:
                continue
            tag = _score(situation_genres, situation_moods, song)
            if tag < MIN_SCORE:
                continue
            sim = None
            if usable and cosine_rank_usable(situation_vector, song.embedding):
                sim = cosine_similarity(situation_vector, song.embedding)
            out.append((song, tag, sim))
        return out

    def vectors_for(self, tracks: list[tuple[str, str]]) -> list[tuple[int, tuple[float, ...]]]:
        """(가수, 제목) 목록 중 곡 풀에 벡터가 있는 곡의 (목록 위치, 벡터).

        가수·제목으로만 찾는다 — 주변 인기곡은 Spotify ID 없이 제목·가수만 온다.
        곡 풀에 없는 곡은 빠진다.
        """
        by_key = {
            _song_key(s.artist, s.title): s.embedding
            for s in self._songs
            if s.embedding is not None
        }
        found = []
        for i, (artist, title) in enumerate(tracks):
            embedding = by_key.get(_song_key(artist, title))
            if embedding is not None:
                found.append((i, embedding))
        return found

    def by_genre(self, genre: str) -> list[PoolSong]:
        """그 장르가 주 장르든 보조 장르든 걸린 곡 전부 — 기능6 곡 대결용."""
        return [song for song in self._songs if genre in song.genres]


def score_tags(
    situation_genres: tuple[str, ...],
    situation_moods: tuple[str, ...],
    song_genres: tuple[str, ...],
    song_moods: tuple[str, ...],
) -> float:
    """상황 태그와 곡 태그가 얼마나 겹치는지. 곡 풀과 저장된 곡에 같은 기준을 쓴다."""
    genre_score = 1.0 if set(song_genres) & set(situation_genres) else 0.0
    mood_overlap = len(set(situation_moods) & set(song_moods))
    mood_score = mood_overlap / len(situation_moods) if situation_moods else 0.0
    return GENRE_WEIGHT * genre_score + MOOD_WEIGHT * mood_score


def _score(
    situation_genres: tuple[str, ...], situation_moods: tuple[str, ...], song: PoolSong
) -> float:
    return score_tags(situation_genres, situation_moods, song.genres, song.moods)


def cosine_rank_usable(vector: list[float], embedding: tuple[float, ...] | None) -> bool:
    return embedding is not None and len(embedding) == len(vector)


def cosine_similarity(vector: list[float], embedding: tuple[float, ...] | None) -> float:
    """코사인 유사도(두 벡터 모두 길이 1이라 내적과 같다). 비교할 수 없으면 맨 뒤(-1)."""
    if embedding is None or not cosine_rank_usable(vector, embedding):
        return -1.0
    return sum(a * b for a, b in zip(vector, embedding, strict=True))


def norm_artist(text: str) -> str:
    return _norm(text)


def _norm(text: str) -> str:
    # "RESCENE(리센느)"와 "리센느", 띄어쓰기 차이를 같은 이름으로 본다
    return re.sub(r"\(.*?\)|\s", "", text).lower()


def _song_key(artist: str, title: str) -> tuple[str, str]:
    return _norm(artist), _norm(title)


def _vector(song: dict, key: str) -> np.ndarray | None:
    """벡터 칸을 float32 배열로. 곡 4천 개 × 벡터 2개를 파이썬 float 튜플로 들면 메모리가 크다."""
    vector = song.get(key)
    return np.asarray(vector, dtype=np.float32) if vector else None


def load_song_pool(path: Path) -> SongPool | None:
    """곡 풀 파일을 읽는다. 없거나 깨졌으면 None — 그때는 LLM 곡만으로 추천한다."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        songs = []
        for s in data["songs"]:
            if not s["genres"]:
                continue
            lyric = _vector(s, "lyric_embedding")
            description = _vector(s, "description_embedding")
            songs.append(
                PoolSong(
                    artist=s["artist"],
                    title=s["title"],
                    genres=tuple(s["genres"]),
                    moods=tuple(s.get("moods", ())),
                    popularity=int(s.get("popularity", 0)),
                    # 예전 방식(vectors_for 등)은 가사 벡터 → 설명 벡터 순으로 있는 것을 쓴다
                    embedding=lyric if lyric is not None else description,
                    lyric_embedding=lyric,
                    description_embedding=description,
                    song_id=str(s.get("song_id") or ""),
                    hub=float(s.get("hub") or 0.0),
                )
            )
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("곡 풀 파일(%s)이 없거나 깨져 있어요. LLM 곡만으로 추천해요.", path)
        return None
    return SongPool(songs)
