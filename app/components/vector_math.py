"""벡터 유사도 계산 — 기능3 태깅·재랭킹, 기능1 히스토리 매칭이 공유."""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalized_mean(vectors: list[list[float]]) -> list[float]:
    """벡터마다 길이를 1로 맞춘 뒤 평균 내고, 평균도 길이 1로 맞춘다.

    CLIP 문장 여러 개를 하나의 태그 벡터로 합칠 때 쓴다. 길이를 먼저 맞추지 않으면
    길이가 긴 문장 벡터가 평균을 끌고 간다.
    """
    if not vectors:
        raise ValueError("평균 낼 벡터가 없어요.")
    unit = []
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        unit.append([x / norm for x in vector] if norm else list(vector))
    mean = [sum(column) / len(unit) for column in zip(*unit, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean))
    return [x / norm for x in mean] if norm else mean
