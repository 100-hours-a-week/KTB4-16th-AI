"""사용자 취향 벡터 — 상황 벡터에 일정 비율로 섞어서 곡을 고를 때 개인화를 반영한다.

Claude에게 "취향을 30% 반영해"라고 써도 숫자 비율은 지켜지지 않는다. 그래서 비율은 코드에서,
곡을 고르는 질문 벡터에 직접 섞는다:  질문 = (1 - w) × 상황 + w × 내 취향  (그 뒤 길이 1로).
내 취향 벡터는 이 사용자가 저장한 곡들의 분위기 벡터(record_embeddings.music_embedding) 평균이다.
w는 설정 personal_taste_weight — 0이면 끈다. 정답을 알 데이터가 아직 없어서 초기값은 같은
상황 세트로 비교해 정한다 (config 주석 참고).
"""

import math


def _normalize(vector: list[float]) -> list[float] | None:
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return None
    return [x / length for x in vector]


def mean_vector(
    vectors: list[list[float]], weights: list[float] | None = None
) -> list[float] | None:
    """같은 길이의 벡터들의 (가중) 평균 (길이 1로 맞춤). 비었거나 길이가 섞이면 None."""
    if not vectors or len({len(v) for v in vectors}) != 1:
        return None
    if weights is None or len(weights) != len(vectors):
        weights = [1.0] * len(vectors)
    dim = len(vectors[0])
    total = [sum(w * v[i] for w, v in zip(weights, vectors, strict=True)) for i in range(dim)]
    return _normalize(total)


def blend_taste(situation: list[float], taste: list[float] | None, weight: float) -> list[float]:
    """상황 벡터에 취향 벡터를 weight 비율로 섞는다. 섞을 수 없으면 상황 벡터 그대로."""
    if taste is None or weight <= 0 or len(taste) != len(situation):
        return situation
    weight = min(weight, 1.0)
    mixed = [(1 - weight) * s + weight * t for s, t in zip(situation, taste, strict=True)]
    return _normalize(mixed) or situation
