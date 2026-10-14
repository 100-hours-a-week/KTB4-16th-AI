"""기능5 기억 검색 — 찾은 자물쇠로 답변 문장을 만든다 (1단계: 템플릿, LLM 없음).

찾은 기록의 값(날짜·장소·곡)만 그대로 쓴다. 검색으로 확정되지 않은 내용은 문장에 들어갈 수 없다 —
"검색과 생성의 분리, 확정되지 않은 내용은 언급 금지"라는 설계 원칙을 문장 틀로 지킨다.
LLM 문장은 V3에서 이 템플릿 위에 얹고, 인용 검증에 실패하면 이 템플릿으로 돌아온다.
"""

from collections.abc import Sequence

from app.db.repositories.memory_record_repository import MemoryRecord

NO_RESULT_TEXT = "찾으시는 기록이 없어요. 다른 말로 다시 물어봐 주세요."


def compose_answer(records: Sequence[MemoryRecord]) -> str:
    if not records:
        return NO_RESULT_TEXT

    lines = [f"비슷한 기록 {len(records)}개를 찾았어요."]
    for number, record in enumerate(records, start=1):
        parts = [_format_date(record), record.place_name, f"{record.artist_name} - {record.title}"]
        lines.append(f"{number}. " + " · ".join(p for p in parts if p))
    return "\n".join(lines)


def _format_date(record: MemoryRecord) -> str:
    d = record.created_at
    return f"{d.year}년 {d.month}월 {d.day}일"
