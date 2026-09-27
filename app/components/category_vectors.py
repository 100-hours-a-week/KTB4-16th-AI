"""고정 카테고리 라벨 벡터 캐싱.

mood_tags.yaml의 태그는 고정이라, 프로세스 켜져 있는 동안 CLIP 텍스트
인코딩을 한 번만 하고 메모리에 들고 있는다. 요청마다 다시 인코딩하면
사진 1장 볼 때마다 태그 개수만큼 CLIP을 더 부르게 되어 낭비다(429 유발 사례:
실측 중 태그 12개를 매번 동시 호출해 Replicate 요청 제한에 걸렸음).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.clients.clip_client import ClipClient

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

    def __init__(self, clip: ClipClient, path: Path = DEFAULT_TAGS_PATH):
        self._clip = clip
        self._path = path
        self._tags: list[MoodTag] = []

    @property
    def loaded(self) -> bool:
        return bool(self._tags)

    @property
    def tags(self) -> list[MoodTag]:
        return self._tags

    async def load(self) -> None:
        labels = load_tag_labels(self._path)
        vectors = await self._clip.encode_text([clip_label for clip_label, _ in labels])
        self._tags = [
            MoodTag(clip_label=clip_label, ko=ko, vector=vector)
            for (clip_label, ko), vector in zip(labels, vectors, strict=True)
        ]
