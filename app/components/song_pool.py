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
    ) -> list[SongCandidate]:
        """상황에 맞는 곡 count개. 맞는 곡이 모자라면 있는 만큼만 (빈 목록일 수 있다)."""
        if count <= 0:
            return []
        excluded = {_song_key(s.artist, s.title) for s in exclude or []}
        scored = [
            (score, song)
            for song in self._songs
            if _song_key(song.artist, song.title) not in excluded
            and (score := _score(situation_genres, situation_moods, song)) >= MIN_SCORE
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
            )
            for s in data["songs"]
            if s["genres"] and s["moods"]
        ]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("곡 풀 파일(%s)이 없거나 깨져 있어요. LLM 곡만으로 추천해요.", path)
        return None
    return SongPool(songs)
