"""추천 이유 한 줄 — 기능1·3 응답의 recommendReason.

모델이 쓴 장면 문장을 그대로 보여 줬더니 엉뚱한 문장이 나와서(dev 확인), 우리가 가진 값으로만
문장을 만든다. 들어가는 값은 상황(날씨·계절·평일/주말·시간대), 실제로 고른 곡의 장르, 취향
장르를 반영했는지 뿐이라 이유와 곡이 어긋나지 않는다. 기분 단어(잔잔한·신나는)는 근거가 되는 값이
없어서 넣지 않는다.

  기능1: "비 오는 겨울 평일 저녁이라 발라드·R&B/Soul 위주로 골라봤어요.
          좋아하시는 장르(발라드)도 반영했어요."
  기능3: "사진 분위기에 맞춰 인디음악·포크/블루스 위주로 골라봤어요."
"""

from collections import Counter

from app.components.music_tags import GENRE_VOCAB

SHOWN_GENRES = 2
SHOWN_TASTE = 3


def top_genres(genres: list[str], limit: int = SHOWN_GENRES) -> list[str]:
    """많이 나온 장르 순(같으면 먼저 나온 순). 13개 장르 목록에 없는 값(공통 등)은 뺀다."""
    known = [g for g in genres if g in GENRE_VOCAB]
    counts = Counter(known)
    first = {g: known.index(g) for g in counts}
    return sorted(counts, key=lambda g: (-counts[g], first[g]))[:limit]


def context_reason(when: str, genres: list[str], taste: list[str] | None = None) -> str:
    """when은 "비 오는 겨울 평일 저녁"처럼 상황을 이은 말(시간대로 끝나서 '이라'가 붙는다)."""
    picked = top_genres(genres)
    head = f"{when}이라 " if when else ""
    body = f"{'·'.join(picked)} 위주로 골라봤어요." if picked else "어울리는 곡으로 골라봤어요."
    text = head + body
    if taste:
        text += f" 좋아하시는 장르({'·'.join(taste[:SHOWN_TASTE])})도 반영했어요."
    return text


def photo_reason(genres: list[str]) -> str:
    picked = top_genres(genres)
    if picked:
        return f"사진 분위기에 맞춰 {'·'.join(picked)} 위주로 골라봤어요."
    return "사진 분위기에 어울리는 곡으로 골라봤어요."
