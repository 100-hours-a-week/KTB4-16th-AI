"""기능5 기억 검색 — 답변 문장을 만든다.

자물쇠 목록은 문장에 넣지 않는다. 프론트가 응답의 records로 카드 목록을 보여주고(채팅에는 앞 5건,
"전체보기" 임시 창에서 최대 50건), 일부만 맞는 자물쇠는 suggestedRecords로 따로 보내서
"이런 자물쇠를 찾으시는 건 아닌가요?" 토글 안에 카드로 보여주기 때문이다.
문장에는 아래 둘만 들어간다.

1. 앞머리 — 질문한 조건과 결과 유무(있음·없음·많아서 일부만). **건수 숫자는 쓰지 않는다** (건수는
   프론트가 records.length로 보여준다). LLM이 쓰고(answer_headline), 실패하거나 숫자가 섞이면 이
   모듈의 템플릿(template_headline)으로 대신한다.
2. 안내 — 사전에 없는 장소 때문에 적용하지 못한 조건 등. 코드가 정한 문장을 그대로 붙인다.

문장에 자물쇠 하나하나의 날짜·장소·곡이 들어가지 않으므로, 대화를 저장해 두고 나중에 자물쇠를
삭제해도 저장된 문장에 그 자물쇠의 정보가 남지 않는다.
"""

from collections.abc import Sequence

NO_RESULT_TEXT = "찾으시는 자물쇠가 없어요. 다른 말로 다시 물어봐 주세요."
UNKNOWN_PLACE_HINT = "동이나 구 이름(예: 성수동, 성동구)으로 물어보시면 더 정확하게 찾을 수 있어요."


def template_headline(
    labels: Sequence[str], *, found: int, total: int | None = None, suggestions: int = 0
) -> str:
    """LLM 없이 만드는 앞머리 문장.

    labels: 질문에서 읽어 낸 조건 이름들 (없으면 조건 없는 질문).
    found: 결과(records)로 보내는 자물쇠 수. 문장에는 숫자로 쓰지 않고 있는지 없는지만 쓴다.
    total: 조건에 모두 맞는 자물쇠 전체 수. found보다 크면 일부만 보내는 것이다.
    suggestions: 제안(suggestedRecords)으로 보내는 일부만 맞는 자물쇠 수.
    """
    if found == 0 and suggestions == 0:
        return NO_RESULT_TEXT
    if not labels:
        return "비슷한 자물쇠를 찾았어요."

    conditions = ", ".join(labels)
    if found == 0:
        return f"{conditions} 조건에 모두 맞는 자물쇠는 없어요."
    total = max(total or found, found)
    if total > found:
        return f"{conditions} 조건에 맞는 자물쇠가 많아서 일부만 보여드려요."
    return f"{conditions} 조건에 맞는 자물쇠를 찾았어요."


def compose_answer(headline: str, *, notices: Sequence[str] = ()) -> str:
    """앞머리 + (있으면) 안내 문장을 한 문자열로 합친다.

    notices는 결과 뒤에 덧붙이는 안내 문장들이다(예: 사전에 없는 장소라 적용 못 한 조건).
    찾은 자물쇠가 없어도 안내는 붙는다.
    """
    lines = [headline]
    if notices:
        lines += ["", *notices]
    return "\n".join(lines)


def unknown_place_notices(found: Sequence[str], missing: Sequence[str]) -> list[str]:
    """사전에 없는 장소 말에 대한 안내 문장들.

    found: 코멘트에서 찾아 조건으로 쓴 말. missing: 어느 코멘트에도 없어서 적용하지 못한 말.
    """
    notices = []
    if found:
        notices.append(
            f"{_quoted(found)}{_topic(found[-1])} 장소 목록에 없는 곳이라서, "
            "기록에 적어 둔 코멘트에서 찾았어요."
        )
    if missing:
        notices.append(
            f"{_quoted(missing)}{_topic(missing[-1])} 아직 모르는 장소라서 "
            f"위치 조건은 적용하지 못했어요. {UNKNOWN_PLACE_HINT}"
        )
    return notices


def _quoted(words: Sequence[str]) -> str:
    return ", ".join(f"'{w}'" for w in words)


def _topic(word: str) -> str:
    """주제 조사: 받침이 있으면 은, 없으면 는. 한글로 안 끝나면 은(는)."""
    last = word[-1]
    if not "가" <= last <= "힣":
        return "은(는)"
    return "은" if (ord(last) - ord("가")) % 28 else "는"
