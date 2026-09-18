"""키 없이 로컬에서 파이프라인을 돌리기 위한 결정적 가짜 벡터."""

import hashlib
import math
import random


def fake_vector(text: str, dim: int) -> list[float]:
    """같은 입력이면 항상 같은 단위 벡터를 돌려준다."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]
