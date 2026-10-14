"""사전에 없는 장소 표현 찾기 — 기능5 기억 검색 (순수 함수).

명소 사전에 없는 "광교산에서" 같은 말은 사전으로는 좌표를 알 수 없다. 그렇다고 조용히 무시하면
사용자는 위치 조건이 적용된 줄 알게 되므로, 이런 말을 찾아서 코멘트에 그 글자가 있는지 따로 본다
(condition_scoring의 place_text 조건).

이름의 끝말(산·공원·역·시장 …)로 장소인지 짐작한다. 끝말만으로 된 일반 낱말("공원", "도서관")과
"지역"처럼 끝말이 우연히 같은 낱말은 장소로 보지 않는다. 모든 장소를 찾지는 못한다.
"""

import re

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")

# 긴 조사부터 뗀다. 떼고 남는 글자가 2자 미만이면 떼지 않는다 ("비도"의 "도")
_PARTICLES = (
    "에서는",
    "에서도",
    "에서",
    "에는",
    "에도",
    "으로",
    "이랑",
    "처럼",
    "까지",
    "부터",
    "보다",
    "에",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "도",
    "와",
    "과",
    "랑",
    "로",
    "만",
)

# 긴 끝말부터 확인한다
_SUFFIXES = (
    "해수욕장",
    "대학교",
    "백화점",
    "터미널",
    "캠퍼스",
    "수목원",
    "박물관",
    "미술관",
    "도서관",
    "리조트",
    "공원",
    "시장",
    "호수",
    "해변",
    "타워",
    "계곡",
    "폭포",
    "마을",
    "거리",
    "공항",
    "산",
    "역",
    "숲",
    "섬",
    "궁",
)
# 끝말이 우연히 같은 낱말
_NOT_PLACES = frozenset(
    {"지역", "번역", "통역", "길거리", "먹거리", "볼거리", "놀거리", "마실거리", "출산", "재산"}
)
_MIN_LENGTH = 3  # 끝말 앞에 글자가 있어야 하고, 두 글자짜리 낱말("계산", "자산")은 제외


def strip_particle(word: str) -> str:
    """단어 끝의 조사를 뗀다 ("광교산에서" → "광교산")."""
    for particle in _PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= 2:
            return word[: -len(particle)]
    return word


def find_unknown_places(remaining_text: str) -> list[str]:
    """조건으로 잡히고 남은 질문에서 장소 같은 말을 찾는다. 순서를 지키고 중복은 뺀다."""
    found: list[str] = []
    for word in _WORD.findall(remaining_text):
        base = strip_particle(word)
        if len(base) < _MIN_LENGTH or base in _NOT_PLACES or base in found:
            continue
        if any(base.endswith(s) and len(base) > len(s) for s in _SUFFIXES):
            found.append(base)
    return found
