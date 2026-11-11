"""사용자의 취향 장르 — 온보딩에서 고른 장르 + 자물쇠에 저장한 곡들의 장르 (기능1).

규칙 기반이고 모델이 없다. 두 출처를 합친다:
  1. 자물쇠 곡 장르(track_moods.genre) 중 많이 나온 상위 LOCK_GENRE_LIMIT개
     (장르 태그가 없는 곡은 세지 않는다 — 곡마다 장르를 아는 곡만 센다)
  2. 온보딩 선호 장르(users.preferred_genres, 1~3개)
둘 다 없으면 빈 목록 = 취향으로 거르지 않는다. 읽기에 실패해도 빈 목록으로 계속한다
(취향은 있으면 좋은 신호라 추천 전체를 실패시키지 않는다).
"""

import logging
from collections import Counter
from collections.abc import Awaitable, Callable

from app.components.music_tags import GENRE_VOCAB
from app.components.vector_search import RecordSearch

logger = logging.getLogger("muro.taste_genres")

LOCK_GENRE_LIMIT = 3

OnboardingReader = Callable[[int], Awaitable[list[str]]]


async def no_onboarding(user_id: int) -> list[str]:
    """백엔드 DB가 없는 로컬·테스트 환경 — 온보딩을 읽을 수 없다."""
    return []


async def taste_genres(
    user_id: int, onboarding: OnboardingReader, records: RecordSearch
) -> list[str]:
    genres: list[str] = []
    try:
        counts = Counter(
            g for g in (await records.genres_by_record(user_id)).values() if g in GENRE_VOCAB
        )
        genres += [genre for genre, _ in counts.most_common(LOCK_GENRE_LIMIT)]
    except Exception:
        logger.warning("자물쇠 곡 장르를 읽지 못함 — 온보딩 장르만 사용", exc_info=True)
    try:
        genres += [g for g in await onboarding(user_id) if g in GENRE_VOCAB and g not in genres]
    except Exception:
        logger.warning("온보딩 장르를 읽지 못함", exc_info=True)
    return genres
