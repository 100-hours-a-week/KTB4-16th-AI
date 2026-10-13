"""기능5 기억 검색 — 찾은 자물쇠로 답변 문장을 만든다 (템플릿, LLM 없음).

찾은 기록의 값(날짜·장소·곡)만 그대로 쓴다. 검색으로 확정되지 않은 내용은 문장에 들어갈 수 없다 —
"검색과 생성의 분리, 확정되지 않은 내용은 언급 금지"라는 설계 원칙을 문장 틀로 지킨다.
LLM 문장은 V3에서 이 템플릿 위에 얹고, 인용 검증에 실패하면 이 템플릿으로 돌아온다.

질문에 조건(장소·계절 등)이 있으면 모든 조건에 맞는 기록과, 일부만 맞는 기록을 나눠서 쓴다.
일부만 맞는 기록은 "이런 기록을 찾으시는 건 아닌가요?"로 제안하고, 어떤 조건이 맞았는지 적는다.
제안은 문장으로만 나간다 — 응답의 matchedRecords에는 모든 조건에 맞는 기록만 들어간다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.db.repositories.memory_record_repository import MemoryRecord

NO_RESULT_TEXT = "찾으시는 기록이 없어요. 다른 말로 다시 물어봐 주세요."
SUGGESTION_HEADER = "이런 기록을 찾으시는 건 아닌가요?"
SUGGESTION_HEADER_WITHOUT_EXACT = "조건에 모두 맞는 기록은 없어요. " + SUGGESTION_HEADER
UNKNOWN_PLACE_HINT = "동이나 구 이름(예: 성수동, 성동구)으로 물어보시면 더 정확하게 찾을 수 있어요."


@dataclass(frozen=True)
class Suggestion:
    record: MemoryRecord
    matched: tuple[str, ...]  # 이 기록이 맞은 조건의 이름


def compose_answer(
    records: Sequence[MemoryRecord],
    suggestions: Sequence[Suggestion] = (),
    *,
    conditioned: bool = False,
    notices: Sequence[str] = (),
) -> str:
    """conditioned는 질문에 조건이 있었는지. 없으면 비슷한 기록만 한 목록으로 쓴다.

    notices는 결과 뒤에 덧붙이는 안내 문장들이다(예: 사전에 없는 장소라 적용 못 한 조건).
    찾은 기록이 없어도 안내는 붙는다.
    """
    lines: list[str] = []
    if not records and not suggestions:
        lines.append(NO_RESULT_TEXT)

    if records:
        noun = "조건에 모두 맞는 기록" if conditioned else "비슷한 기록"
        lines.append(f"{noun} {len(records)}개를 찾았어요.")
        lines += _numbered(r for r in records)
    if suggestions:
        if lines:
            lines.append("")
        lines.append(SUGGESTION_HEADER if records else SUGGESTION_HEADER_WITHOUT_EXACT)
        lines += _numbered(
            (s.record for s in suggestions), notes=[_matched_note(s.matched) for s in suggestions]
        )
    if notices:
        lines.append("")
        lines += notices
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


def _numbered(records, notes: Sequence[str] = ()) -> list[str]:
    lines = []
    for number, record in enumerate(records, start=1):
        parts = [_format_date(record), record.place_name, f"{record.artist_name} - {record.title}"]
        line = f"{number}. " + " · ".join(p for p in parts if p)
        if notes and notes[number - 1]:
            line += f" ({notes[number - 1]})"
        lines.append(line)
    return lines


def _matched_note(matched: Sequence[str]) -> str:
    if not matched:
        return ""
    return ", ".join(f"'{label}'" for label in matched) + "만 맞아요"


def _format_date(record: MemoryRecord) -> str:
    d = record.created_at
    return f"{d.year}년 {d.month}월 {d.day}일"
