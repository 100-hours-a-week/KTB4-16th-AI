"""질문에서 이름(명소·지역 등)을 찾는 공통 규칙 — 기능5 기억 검색 (순수 함수).

이름이 더 긴 단어의 일부일 때 잘못 잡지 않으려고 앞뒤를 본다.
- 앞: 한글이 바로 붙어 있으면 안 된다 ("스이대" 속 "이대"는 아니다)
- 뒤: 끝이거나, 한글이 아닌 글자이거나, 조사·위치말이어야 한다
  ("이대로"의 "이대"나 "중대한"의 "중대"는 아니고, "이대에서"·"서울숲 근처"는 맞다)
이름 안의 공백은 있어도 없어도 된다 ("홍대 앞"과 "홍대앞").
"""

import re

# 이름 바로 뒤에 올 수 있는 조사·위치말.
# "로/으로"는 일부러 뺐다 ("이대로"가 "이대"로 잡히는 걸 막으려고).
FOLLOWING = (
    r"(?=$|[^가-힣]|에서|에|의|은|는|이|가|을|를|도|와|과|랑|쪽|근처|부근|주변|앞|뒤|쯤|까지|부터)"
)
NOT_AFTER_HANGUL = r"(?<![가-힣])"


def squash(name: str) -> str:
    """비교용으로 공백을 없애고 소문자로 맞춘다 (중복 검사에 쓴다)."""
    return "".join(name.split()).lower()


def name_pattern(name: str) -> re.Pattern[str]:
    chars = [re.escape(c) for c in name if not c.isspace()]
    if not chars:
        raise ValueError("이름이 비어 있어요.")
    return re.compile(NOT_AFTER_HANGUL + r"\s*".join(chars) + FOLLOWING, re.IGNORECASE)


def substring_pattern(name: str) -> re.Pattern[str]:
    """앞뒤 글자를 따지지 않고 이름이 들어 있기만 하면 잡는다 (공백·대소문자는 무시).

    사전에 없는 "광교산" 같은 긴 말을 코멘트에서 찾을 때 쓴다 — "광교산입구"도 잡혀야 해서
    name_pattern의 뒤 글자 규칙을 쓰지 않는다. 짧은 별칭("이대")에는 쓰지 말 것.
    """
    chars = [re.escape(c) for c in name if not c.isspace()]
    if not chars:
        raise ValueError("이름이 비어 있어요.")
    return re.compile(r"\s*".join(chars), re.IGNORECASE)


def blank_out(text: str, start: int, end: int) -> str:
    """찾은 글자를 같은 길이의 공백으로 바꾼다 — 같은 글자를 다른 규칙이 또 잡지 못하게."""
    return text[:start] + " " * (end - start) + text[end:]
