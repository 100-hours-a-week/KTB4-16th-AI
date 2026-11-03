"""모델이 판단할 근거가 없는 아주 짧은 입력을 모델 없이 정상으로 보는 규칙.

2026-11-03 측정: 운영 모델(kcelectra-v1-stage3)은 학습 데이터에 짧은 정상 채팅이 거의 없어서
"ㅋㅋ", "?", "헐" 같은 입력에 0.55 근처를 찍고 "ㅇ"(0.84)·"ㅂㅂ"(0.94)처럼 확신하고 틀리기도 했다.
짧은말 시험지 정상 110개 중 기준값 0.5에서 51개를 막았고, 이 규칙을 넣으면 18개로 줄었다
(음악채팅·변형욕설 시험지 결과는 그대로).

막을 초성 목록을 쓰면 빠지는 게 생겨서(ㄷㅊ·ㄸㄹㅇ·ㄱㅅㄲ가 빠졌다) 반대로 안전한 것만 건너뛴다.
- 웃음·울음·기호·숫자·이모지만 있는 입력 (ㅋㅋ, ㅎㅎ, ㅠㅠ, ?, 👍)
- 자주 쓰는 초성 대답 (ㅇㅇ, ㅇㅋ, ㄱㄱ …)
- 공백을 빼고 한 글자인 입력 (ㅇ, 네, 오). 손가락 욕 ㅗ와 ㅄ은 제외
그 밖의 초성 묶음과 두 글자 이상의 말은 모델이 본다.
"""

import re

LAUGH_OR_SYMBOL = re.compile(r"^[ㅋㅎㅠㅜㅡ0-9\W_]*$")
SAFE_JAMO = frozenset(
    "ㅇㅇ ㅇㅋ ㅇㅈ ㅇㅎ ㄱㄱ ㄱㄱㅆ ㄴㄴ ㄹㅇ ㄷㄷ ㅎㅇ ㅂㅂ ㅂㅇ "
    "ㄳ ㄱㅅ ㅊㅋ ㅈㅅ ㅇㅋㅇㅋ ㄹㅇㅋㅋ ㅇㄷ ㄴㄴㄴ ㅇㅇㅇ".split()
)
UNSAFE_CHARS = frozenset({"ㅗ", "ㅄ"})
# "ㅇㅇ?", "ㄱㄱㅋㅋ"처럼 끝에 붙은 웃음·기호는 떼고 본다
TRAILING = re.compile(r"[ㅋㅎㅠㅜ?!.~,]+$")


def skip_model(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if any(c in UNSAFE_CHARS for c in compact):
        return False
    if LAUGH_OR_SYMBOL.match(compact) or len(compact) == 1:
        return True
    return TRAILING.sub("", compact) in SAFE_JAMO
