"""RECAP 회고 문장의 자동 채점 기준 (좋은 회고 글의 "반드시 지킬 것").

v7 프롬프트(prompt_v7.py, 한때 서비스 프롬프트)의 규칙을 코드로 옮긴 것이다. 서비스는 v9로 바뀌었다.
위반이 하나라도 있으면 그 문장은 학습용 정답으로 쓰지 않는다. 톤(다정하고 담백한지), 패턴을 잘
짚었는지 같은 사람이 봐야 하는 기준은 README.md의 "사람이 볼 것"을 따른다.

check()는 위반 목록을 돌려준다. 빈 목록이면 통과다.
"""

import re

from scripts.recap_distill import pools

TEXT_LIMIT = (
    100  # 백엔드 컬럼 ai_recap_text가 VARCHAR(100) — report_summarizer.AI_RECAP_TEXT_LIMIT과 같다
)
RELATION_WORDS = ("친구", "가족", "연인", "애인", "부모님", "남자친구", "여자친구")
COMMENT_QUOTE_MIN_CHARS = 8  # 이보다 긴 코멘트가 그대로 들어 있으면 "인용"으로 본다

_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")
_SCORE_NUMBER = re.compile(r"-?\d+\s*점")
_COMMENT_LINE = re.compile(r"코멘트: (.+)$")


def _comments_in(prompt: str) -> list[str]:
    return [m.group(1).strip() for line in prompt.splitlines() if (m := _COMMENT_LINE.search(line))]


def check(text: str, prompt: str) -> list[str]:
    """text(생성된 회고 문장)가 prompt(입력)에 대해 어기는 규칙의 이름들."""
    text = text.strip()
    if not text:
        return ["비어 있음"]

    violations = []
    if not text.startswith("당신은"):
        violations.append("'당신은'으로 시작하지 않음")
    if len(text) > TEXT_LIMIT:
        violations.append(f"{TEXT_LIMIT}자 초과")
    if "\n" in text or len(_SENTENCE_END.findall(text.rstrip(".!?… "))) > 0:
        violations.append("한 문장이 아님")
    if "기분점수" in text or _SCORE_NUMBER.search(text):
        violations.append("기분점수를 언급함")

    invented = [
        name for name in pools.ARTISTS + pools.PLACES if name in text and name not in prompt
    ]
    if invented:
        violations.append(f"입력에 없는 이름을 말함: {', '.join(invented)}")

    relations = [w for w in RELATION_WORDS if w in text and w not in prompt]
    if relations:
        violations.append(f"입력에 없는 관계를 추측함: {', '.join(relations)}")

    quoted = [c for c in _comments_in(prompt) if len(c) >= COMMENT_QUOTE_MIN_CHARS and c in text]
    if quoted:
        violations.append("코멘트를 그대로 인용함")
    return violations
