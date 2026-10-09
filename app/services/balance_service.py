"""기능6 오늘의 질문 생성 — 운영·배치용 (사용자 요청으로 호출되지 않음).

① 최근 자물쇠에서 이 장르 그룹의 가수·무드·장소·날씨 통계를 모은다
② 통계를 재료로 LLM이 A vs B 취향 질문을 만든다
③ 형식이 맞는 질문만 돌려준다. 질문 저장·questionId 발급은 백엔드가 한다
"""

import logging

from app.components.question_generator import QuestionGenerator
from app.db.repositories.balance_stats_repository import GenreStats, GenreStatsSource
from app.exceptions import LLMError
from app.schemas.balance import (
    BalanceQuestion,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
)

logger = logging.getLogger("muro.balance")


class BalanceService:
    def __init__(self, stats: GenreStatsSource, generator: QuestionGenerator):
        self._stats = stats
        self._generator = generator

    async def generate_questions(self, req: QuestionGenerateRequest) -> QuestionGenerateResponse:
        try:
            stats = await self._stats.get(req.music_genre)
        except Exception:
            # 통계는 질문을 더 우리 서비스답게 만드는 재료일 뿐이라, DB를 못 읽어도 질문은 만든다
            logger.exception("오늘의 질문 통계 조회 실패 — 통계 없이 생성")
            stats = GenreStats.empty()

        questions = await self._generator.generate(req.music_genre, stats, req.count)
        if not questions:
            raise LLMError("질문을 만들지 못했어요.")
        return QuestionGenerateResponse(
            questions=[
                BalanceQuestion(question=q, option_a=a, option_b=b, music_genre=req.music_genre)
                for q, a, b in questions
            ]
        )
