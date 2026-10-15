"""고정 카테고리 라벨 벡터 캐싱.

mood_tags.yaml의 태그는 고정이라, 프로세스 켜져 있는 동안 CLIP 텍스트
인코딩을 한 번만 하고 메모리에 들고 있는다. 요청마다 다시 인코딩하면
사진 1장 볼 때마다 태그 개수만큼 CLIP을 더 부르게 되어 낭비다(429 유발 사례:
실측 중 태그 12개를 매번 동시 호출해 Replicate 요청 제한에 걸렸음).
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.clients.clip_client import ClipClient
from app.components.tag_vector_file import PrecomputedVectors

logger = logging.getLogger("muro.category_vectors")

# app/components/category_vectors.py 기준 app/mood_tags.yaml
DEFAULT_TAGS_PATH = Path(__file__).resolve().parent.parent / "mood_tags.yaml"


@dataclass(frozen=True)
class MoodTag:
    clip_label: str  # CLIP 인코딩용 (영어)
    ko: str  # 응답 노출용 한국어 표시명
    vector: list[float]


def load_tag_labels(path: Path) -> list[tuple[str, str]]:
    """yaml에서 (clip_label, ko) 쌍만 읽는다. 벡터화 전 순수 파싱 단계라 테스트하기 쉽다."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tags = data.get("tags") or []
    if not tags:
        raise ValueError(f"{path}에 태그가 하나도 없어요.")
    return [(t["clip"], t["ko"]) for t in tags]


class CategoryVectors:
    """CLIP으로 인코딩한 태그 벡터 캐시. `load()`를 앱 시작 시(또는 첫 사용 시) 한 번 호출한다."""

    def __init__(
        self,
        clip: ClipClient,
        path: Path = DEFAULT_TAGS_PATH,
        precomputed: PrecomputedVectors | None = None,
    ):
        self._clip = clip
        self._path = path
        self._precomputed = precomputed
        self._tags: list[MoodTag] = []
        # 요청 여러 개가 동시에 처음 부르면 각자 CLIP을 부르지 않고 한 번만 계산하게 한다
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self._tags)

    @property
    def tags(self) -> list[MoodTag]:
        return self._tags

    async def load(self) -> None:
        async with self._lock:
            if self._tags:  # 기다리는 동안 다른 요청이 이미 끝냈다
                return
            labels = load_tag_labels(self._path)
            vectors = self._precomputed_vectors([clip_label for clip_label, _ in labels])
            if vectors is None:
                logger.warning(
                    "미리 계산한 장소 태그 벡터를 쓸 수 없어 CLIP으로 %d개를 인코딩해요 "
                    "(몇 분 걸릴 수 있음)",
                    len(labels),
                )
                vectors = await self._clip.encode_text([clip_label for clip_label, _ in labels])
            self._tags = [
                MoodTag(clip_label=clip_label, ko=ko, vector=vector)
                for (clip_label, ko), vector in zip(labels, vectors, strict=True)
            ]

    def _precomputed_vectors(self, clip_labels: list[str]) -> list[list[float]] | None:
        """미리 계산한 벡터가 모든 라벨에 있고 CLIP 차원과 맞을 때만 쓴다. 아니면 None."""
        if self._precomputed is None:
            return None
        found = [self._precomputed.scene.get(label) for label in clip_labels]
        dim = getattr(self._clip, "dim", None)
        if any(v is None or (dim is not None and len(v) != dim) for v in found):
            return None
        return found  # type: ignore[return-value]
