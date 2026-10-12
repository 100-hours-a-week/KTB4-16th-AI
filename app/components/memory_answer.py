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


@dataclass(frozen=True)
class Suggestion:
    record: MemoryRecord
    matched: tuple[str, ...]  # 이 기록이 맞은 조건의 이름


def compose_answer(
    records: Sequence[MemoryRecord],
    suggestions: Sequence[Suggestion] = (),
    *,
    conditioned: bool = False,
) -> str:
    """conditioned는 질문에 조건이 있었는지. 없으면 비슷한 기록만 한 목록으로 쓴다."""
    if not records and not suggestions:
        return NO_RESULT_TEXT

    lines: list[str] = []
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
    return "\n".join(lines)


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
