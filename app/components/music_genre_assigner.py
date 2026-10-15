"""기능6 사용자별 취향 투표 그룹(music_genre) 계산 — 규칙 기반, LLM 없음.

1. 저장한 자물쇠 곡들의 장르를 세서 가장 많은 장르 (동률이면 더 최근에 저장한 장르)
2. 장르 정보가 없으면 공통 (온보딩 선호 장르는 백엔드 칸이 생기면 여기에 추가)
3. 그룹 인원이 최소 인원보다 적으면 공통 — 같은 질문에 다른 답을 고른 사람이 모여야
   매칭이 되므로, 사람이 적은 장르로 쪼개면 매칭이 안 된다
4. 공통 그룹마저 최소 인원보다 적으면 가장 큰 장르 그룹에 합친다 — 공통이 너무 작아
   매칭이 안 되는 것보다, 장르가 조금 달라도 사람이 많은 그룹이 낫다
   (자물쇠가 없는 사용자는 여기 안 나오고, 백엔드가 공통으로 처리한다)

결과는 13개 장르 + 공통 중 하나.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from app.components.music_tags import COMMON_GROUP, GENRE_VOCAB

SOURCE_RECORDS = "records"
SOURCE_COMMON = "common"
SOURCE_MERGED = "merged"  # 공통이 작아서 가장 큰 장르 그룹에 합쳐짐


@dataclass(frozen=True)
class SavedTrack:
    user_id: int
    external_track_id: str
    saved_at: datetime


@dataclass(frozen=True)
class Assignment:
    music_genre: str
    source: str


def assign_genres(
    saved: list[SavedTrack], genre_of: dict[str, str], min_group_size: int
) -> dict[int, Assignment]:
    counts: dict[int, Counter] = defaultdict(Counter)
    latest: dict[tuple[int, str], datetime] = {}
    users: set[int] = set()
    for s in saved:
        users.add(s.user_id)
        genre = genre_of.get(s.external_track_id, "")
        if genre not in GENRE_VOCAB:
            continue
        counts[s.user_id][genre] += 1
        key = (s.user_id, genre)
        if key not in latest or s.saved_at > latest[key]:
            latest[key] = s.saved_at

    top: dict[int, str] = {}
    for user_id in users:
        if counts[user_id]:
            top[user_id] = max(
                counts[user_id], key=lambda g: (counts[user_id][g], latest[(user_id, g)])
            )

    group_size = Counter(top.values())
    result: dict[int, Assignment] = {}
    for user_id in users:
        genre = top.get(user_id)
        if genre is None or group_size[genre] < min_group_size:
            result[user_id] = Assignment(COMMON_GROUP, SOURCE_COMMON)
        else:
            result[user_id] = Assignment(genre, SOURCE_RECORDS)

    common = [uid for uid, a in result.items() if a.music_genre == COMMON_GROUP]
    kept = Counter(a.music_genre for a in result.values() if a.music_genre != COMMON_GROUP)
    if common and len(common) < min_group_size and kept:
        biggest = max(kept, key=lambda g: (kept[g], g))
        for uid in common:
            result[uid] = Assignment(biggest, SOURCE_MERGED)
    return result
