"""기능5 기억 검색 — 답변 문장을 만든다.

자물쇠 목록은 문장에 넣지 않는다. 프론트가 응답의 matchedRecords로 카드 목록을 보여주고(앞 5건,
나머지는 "더보기"), 카드를 누르면 해당 자물쇠로 이동하기 때문이다. 문장에는 아래 셋만 들어간다.

1. 앞머리 — 질문한 조건과 찾은 건수. LLM이 쓰고(answer_headline), 실패하면 이 모듈의 템플릿
   (template_headline)으로 대신한다.
2. 제안 — 모든 조건에 맞지는 않지만 일부만 맞는 자물쇠. 카드가 없어서 문장으로만 안내한다.
3. 안내 — 사전에 없는 장소 때문에 적용하지 못한 조건 등. 코드가 정한 문장을 그대로 붙인다.

찾은 값(날짜·장소·곡)을 문장에 쓰는 것은 제안 목록뿐이고, 그 값은 검색으로 확정된 기록의 것이다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.db.repositories.memory_record_repository import MemoryRecord

NO_RESULT_TEXT = "찾으시는 자물쇠가 없어요. 다른 말로 다시 물어봐 주세요."
SUGGESTION_HEADER = "이런 자물쇠를 찾으시는 건 아닌가요?"
UNKNOWN_PLACE_HINT = "동이나 구 이름(예: 성수동, 성동구)으로 물어보시면 더 정확하게 찾을 수 있어요."


@dataclass(frozen=True)
class Suggestion:
    record: MemoryRecord
    matched: tuple[str, ...]  # 이 기록이 맞은 조건의 이름


def template_headline(
    labels: Sequence[str], *, found: int, total: int | None = None, suggestions: int = 0
) -> str:
    """LLM 없이 만드는 앞머리 문장.

    labels: 질문에서 읽어 낸 조건 이름들 (없으면 조건 없는 질문).
    found: 결과(matchedRecords)로 보내는 자물쇠 수.
    total: 조건에 모두 맞는 자물쇠 전체 수. found보다 크면 일부만 보내는 것이다.
    suggestions: 문장 아래에 제안하는 일부만 맞는 자물쇠 수.
    """
    if found == 0 and suggestions == 0:
        return NO_RESULT_TEXT
    if not labels:
        return f"비슷한 자물쇠 {found}건을 찾았어요."

    conditions = ", ".join(labels)
    if found == 0:
        return f"{conditions} 조건에 모두 맞는 자물쇠는 없어요."
    total = max(total or found, found)
    if total > found:
        return (
            f"{conditions} 조건에 맞는 자물쇠는 총 {total}건이에요. "
            f"그중 상위 {found}건을 보여드려요."
        )
    return f"{conditions} 조건에 맞는 자물쇠 {found}건을 찾았어요."


def compose_answer(
    headline: str,
    suggestions: Sequence[Suggestion] = (),
    *,
    notices: Sequence[str] = (),
) -> str:
    """앞머리 + (있으면) 제안 목록 + (있으면) 안내 문장을 한 문자열로 합친다.

    notices는 결과 뒤에 덧붙이는 안내 문장들이다(예: 사전에 없는 장소라 적용 못 한 조건).
    찾은 자물쇠가 없어도 안내는 붙는다.
    """
    lines = [headline]
    if suggestions:
        lines += ["", SUGGESTION_HEADER]
        lines += _numbered(suggestions)
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


def _numbered(suggestions: Sequence[Suggestion]) -> list[str]:
    lines = []
    for number, suggestion in enumerate(suggestions, start=1):
        record = suggestion.record
        parts = [_format_date(record), record.place_name, f"{record.artist_name} - {record.title}"]
        line = f"{number}. " + " · ".join(p for p in parts if p)
        note = _matched_note(suggestion.matched)
        if note:
            line += f" ({note})"
        lines.append(line)
    return lines


def _matched_note(matched: Sequence[str]) -> str:
    if not matched:
        return ""
    return ", ".join(f"'{label}'" for label in matched) + "만 맞아요"


def _format_date(record: MemoryRecord) -> str:
    d = record.created_at
    return f"{d.year}년 {d.month}월 {d.day}일"
