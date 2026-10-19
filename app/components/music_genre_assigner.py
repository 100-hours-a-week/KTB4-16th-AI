"""기능6 사용자별 취향 투표 그룹(music_genre) 계산 — 규칙 기반, LLM 없음.

1. 저장한 자물쇠 곡들의 장르를 세서 가장 많은 장르 (동률이면 더 최근에 저장한 장르)
   온보딩에서 고른 선호 장르(users.preferred_genres)는 장르마다 PREFERRED_WEIGHT점을 더해서
   센다 — 저장한 곡이 적은 새 사용자는 온보딩 장르로, 곡이 쌓이면 실제 저장한 장르로 간다
2. 저장한 곡도 온보딩 장르도 없으면 공통
3. 그룹 인원이 최소 인원보다 적으면 공통 — 같은 질문에 다른 답을 고른 사람이 모여야
   매칭이 되므로, 사람이 적은 장르로 쪼개면 매칭이 안 된다
4. 공통 그룹마저 최소 인원보다 적으면 가장 큰 장르 그룹에 합친다 — 공통이 너무 작아
   매칭이 안 되는 것보다, 장르가 조금 달라도 사람이 많은 그룹이 낫다
   (자물쇠가 없는 사용자는 여기 안 나오고, 백엔드가 공통으로 처리한다)

결과는 13개 장르 + 공통 중 하나.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from app.components.music_tags import COMMON_GROUP, GENRE_VOCAB

SOURCE_RECORDS = "records"
SOURCE_COMMON = "common"
SOURCE_MERGED = "merged"  # 공통이 작아서 가장 큰 장르 그룹에 합쳐짐

# 온보딩 선호 장르 하나는 저장한 곡 PREFERRED_WEIGHT개와 같다 (저장 3곡이면 온보딩과 동점).
# 정답을 알 데이터가 아직 없어서 감으로 정한 초기값이다.
PREFERRED_WEIGHT = 3


@dataclass(frozen=True)
class SavedTrack:
    user_id: int
    external_track_id: str
    saved_at: datetime


@dataclass(frozen=True)
class Assignment:
    music_genre: str
    source: str


def parse_preferred(raw: object) -> list[str]:
    """users.preferred_genres 값을 장르 목록으로. JSON 배열·쉼표로 이은 글·리스트를 모두 받는다.

    13개 장르 목록에 없는 값은 버린다. 순서는 고른 순서를 그대로 둔다(동점일 때 앞이 이긴다).
    """
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except ValueError:
                raw = text.strip("[]").split(",")
        else:
            raw = text.split(",")
    if not isinstance(raw, (list, tuple)):
        return []
    genres: list[str] = []
    for item in raw:
        genre = str(item).strip().strip("\"'")
        if genre in GENRE_VOCAB and genre not in genres:
            genres.append(genre)
    return genres


def assign_genres(
    saved: list[SavedTrack],
    genre_of: dict[str, str],
    min_group_size: int,
    preferred: dict[int, list[str]] | None = None,
) -> dict[int, Assignment]:
    counts: dict[int, Counter] = defaultdict(Counter)
    latest: dict[tuple[int, str], datetime] = {}
    users: set[int] = set()
    preferred_rank: dict[tuple[int, str], int] = {}
    for user_id, genres in (preferred or {}).items():
        for rank, genre in enumerate(genres):
            if genre in GENRE_VOCAB:
                users.add(user_id)
                counts[user_id][genre] += PREFERRED_WEIGHT
                preferred_rank[(user_id, genre)] = rank
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
                counts[user_id],
                key=lambda g: (
                    counts[user_id][g],
                    (1, latest[(user_id, g)]) if (user_id, g) in latest else (0, None),
                    -preferred_rank.get((user_id, g), len(GENRE_VOCAB)),
                ),
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


def assign_all_common(saved: list[SavedTrack]) -> dict[int, Assignment]:
    """모두를 공통 그룹으로 (V2까지 — 장르별로 나누지 않고 한 질문으로 채팅을 연다)."""
    return {s.user_id: Assignment(COMMON_GROUP, SOURCE_COMMON) for s in saved}
