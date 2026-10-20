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

    def to_candidate(self) -> SongCandidate:
        return SongCandidate(
            artist=self.artist, title=self.title, genre=self.genres[0], moods=self.moods
        )


class SongPool:
    def __init__(self, songs: list[PoolSong], rng: random.Random | None = None):
        self._songs = songs
        self._rng = rng or random.Random()

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
        by_key = {_song_key(s.artist, s.title): s.embedding for s in self._songs if s.embedding}
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


def load_song_pool(path: Path) -> SongPool | None:
    """곡 풀 파일을 읽는다. 없거나 깨졌으면 None — 그때는 LLM 곡만으로 추천한다."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        songs = [
            PoolSong(
                artist=s["artist"],
                title=s["title"],
                genres=tuple(s["genres"]),
                moods=tuple(s["moods"]),
                popularity=int(s.get("popularity", 0)),
                embedding=tuple(s["description_embedding"])
                if s.get("description_embedding")
                else None,
            )
            for s in data["songs"]
            if s["genres"] and s["moods"]
        ]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("곡 풀 파일(%s)이 없거나 깨져 있어요. LLM 곡만으로 추천해요.", path)
        return None
    return SongPool(songs)
