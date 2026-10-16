"""기능6 오늘의 질문 — 곡 대결 유형 (LLM 없이, 곡 풀에서 체급 맞춰 두 곡을 뽑는다).

question_generator(취향 질문)와 달리 LLM을 부르지 않는다 — 실제로 존재하는 두 곡을
곡 풀에서 직접 뽑으니 지어낼 염려가 없고, 비용도 0이다. "체급"은 지금은 곡 풀의
popularity(TJ 노래방 차트에 오른 개월 수, 0~100)로 맞춘다. 유튜브 조회수 같은 실제
인기 신호가 준비되면 PoolSong.popularity 자체를 그 값으로 바꾸기만 하면 되고, 이
파일은 안 바뀐다.

곡 풀이 그 장르에서 너무 얇으면(쓸 수 있는 짝이 없음) None을 돌려주고, 그러면
balance_daily_worker가 기존 LLM 취향 질문으로 넘어간다.
"""

import random
from dataclasses import dataclass

from app.components.song_pool import PoolSong, SongPool

PROMPT_VERSION = "song-battle-v1"
# 체급차 — popularity 0~100 기준. TJ 개월 수로는 이 정도면 "둘 다 꽤 익숙한 곡" 묶음.
MAX_POPULARITY_GAP = 20
OPTION_MAX_LEN = 50  # vote_questions.option_a/b 컬럼 크기(String(50))에 맞춤
TEMPLATES = (
    "지금 이 기분에 더 맞는 노래는?",
    "오늘 플레이리스트에 넣고 싶은 곡은?",
    "노래방에서 부르고 싶은 곡은?",
    "친구에게 추천하고 싶은 곡은?",
    "지금 듣고 싶은 노래는?",
)


@dataclass(frozen=True)
class SongBattle:
    question: str
    option_a: str
    option_b: str

    @property
    def pair_key(self) -> frozenset[str]:
        return frozenset({self.option_a, self.option_b})


def generate_battle(
    genre: str,
    pool: SongPool,
    used_pairs: set[frozenset[str]],
    rng: random.Random | None = None,
) -> SongBattle | None:
    """체급이 비슷하고 최근에 안 쓴 두 곡을 뽑는다. 쓸 짝이 없으면 None."""
    rng = rng or random.Random()
    songs = sorted(pool.by_genre(genre), key=lambda s: s.popularity)
    pairs = [
        (a, b)
        for i, a in enumerate(songs)
        for b in songs[i + 1 :]
        if abs(a.popularity - b.popularity) <= MAX_POPULARITY_GAP
        and _option(a) != _option(b)
        and frozenset({_option(a), _option(b)}) not in used_pairs
    ]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    question = rng.choice(TEMPLATES)
    return SongBattle(question=question, option_a=_option(a), option_b=_option(b))


def _option(song: PoolSong) -> str:
    text = f"{song.artist} - {song.title}"
    return text if len(text) <= OPTION_MAX_LEN else text[: OPTION_MAX_LEN - 1] + "…"
