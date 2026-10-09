"""기능6 취향 투표 — 오늘의 질문 생성 (위키 모델 API 설계 6-1).

V2: 질문을 장르 그룹(music_genre)별로 만든다. 백엔드가 users.music_genre와 같은 그룹의
질문을 보여주고, 같은 질문에 다른 답을 고른 사람끼리 매칭한다.
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel

MusicGenreGroup = Literal[
    "발라드",
    "댄스",
    "랩/힙합",
    "R&B/Soul",
    "인디음악",
    "록/메탈",
    "포크/블루스",
    "트로트",
    "POP",
    "일렉트로니카",
    "OST",
    "재즈",
    "J-POP",
    "공통",
]


class QuestionGenerateRequest(CamelModel):
    music_genre: MusicGenreGroup
    count: int = Field(default=5, ge=1, le=10)


class BalanceQuestion(CamelModel):
    # 백엔드 투표 질문 테이블(question, optionA, optionB)과 이름을 맞춘다
    question: str
    option_a: str
    option_b: str
    music_genre: MusicGenreGroup


class QuestionGenerateResponse(CamelModel):
    questions: list[BalanceQuestion]
