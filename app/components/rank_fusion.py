"""검색기 결과 합치기 — 기능5 기억 검색 (순수 함수, DB·외부 호출 없음).

코멘트 벡터와 곡 분위기 벡터는 둘 다 "질문 글 ↔ 글" 코사인 유사도라 값의 범위가 비슷하다.
그래서 순위만 쓰는 RRF 대신 유사도 값을 가중 평균한다(점수 차이가 큰 1등을 살리려는 것).
검색기 하나에만 잡힌 기록은 다른 검색기 몫을 0으로 계산해서, 둘 다에서 잡힌 기록이 위로 간다.

가중치·최소 유사도는 시험지로 정할 초기값이다 — 하드코딩하지 않고 호출하는 쪽이 넘긴다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.components.condition_scoring import RecordScore
from app.components.vector_search import SimilarRecord


@dataclass(frozen=True)
class FusedRecord:
    record_id: int
    score: float  # 0~1, 두 유사도의 가중 평균 (없는 쪽은 0)
    comment_similarity: float | None
    music_similarity: float | None


def fuse(
    comment_hits: Sequence[SimilarRecord],
    music_hits: Sequence[SimilarRecord],
    *,
    comment_weight: float,
    music_weight: float,
    min_similarity: float,
) -> list[FusedRecord]:
    """두 검색 결과를 record_id 기준으로 합쳐 점수 높은 순으로 돌려준다.

    - min_similarity 미만인 항목은 그 검색기 결과에서 먼저 뺀다. 둘 다 미달이면 기록이 사라진다.
    - 점수가 같으면 record_id가 큰(최근) 기록을 앞에 둔다 — 결과 순서가 매번 같게.
    """
    total_weight = comment_weight + music_weight
    if comment_weight < 0 or music_weight < 0 or total_weight <= 0:
        raise ValueError("가중치는 0 이상이고 합이 0보다 커야 해요.")

    comment_by_id = {
        h.record_id: h.similarity for h in comment_hits if h.similarity >= min_similarity
    }
    music_by_id = {h.record_id: h.similarity for h in music_hits if h.similarity >= min_similarity}

    fused = []
    for record_id in comment_by_id.keys() | music_by_id.keys():
        comment = comment_by_id.get(record_id)
        music = music_by_id.get(record_id)
        score = (comment_weight * (comment or 0.0) + music_weight * (music or 0.0)) / total_weight
        fused.append(FusedRecord(record_id, score, comment, music))

    fused.sort(key=lambda f: (-f.score, -f.record_id))
    return fused


@dataclass(frozen=True)
class RankedRecord:
    record_id: int
    score: float  # 최종 점수 (0~1)
    vector_score: float
    condition: RecordScore


@dataclass(frozen=True)
class ConditionRanking:
    exact: list[RankedRecord]  # 질문의 조건을 모두 만족하는 기록
    suggestions: list[RankedRecord]  # 일부만 만족하는 기록 ("이런 걸 찾으시는 게 아닌가요?")


def combine_with_conditions(
    vector: Sequence[FusedRecord],
    conditions: Mapping[int, RecordScore],
    *,
    condition_weight: float,
) -> ConditionRanking:
    """벡터 점수와 조건 점수를 합쳐서, 조건을 모두 만족하는 기록과 일부만 만족하는 기록으로 나눈다.

    최종 점수 = (1 - w) × 벡터 점수 + w × 조건 점수. 조건을 하나도 못 맞춘 기록은 벡터 점수가
    아무리 높아도 결과에 넣지 않는다 — 질문이 조건을 말했는데 상관없는 기록이 섞이지 않게.
    벡터 검색 상위 후보에 못 든 기록(조건으로만 후보가 된 기록)은 벡터 점수 0으로 계산한다.
    """
    if not 0 <= condition_weight <= 1:
        raise ValueError("조건 가중치는 0~1 사이여야 해요.")

    vector_by_id = {f.record_id: f.score for f in vector}
    exact: list[RankedRecord] = []
    suggestions: list[RankedRecord] = []
    for record_id, condition in conditions.items():
        if not condition.matched:
            continue
        vector_score = vector_by_id.get(record_id, 0.0)
        score = (1 - condition_weight) * vector_score + condition_weight * condition.score
        ranked = RankedRecord(record_id, score, vector_score, condition)
        (exact if condition.all_matched else suggestions).append(ranked)

    for group in (exact, suggestions):
        group.sort(key=lambda r: (-r.score, -r.record_id))  # 점수가 같으면 최근 기록 먼저
    return ConditionRanking(exact, suggestions)
