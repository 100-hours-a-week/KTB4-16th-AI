"""다축 태그 정의(axis_tags.yaml)와 태그 벡터 캐시.

태그마다 영어 문장 2~3개를 CLIP 텍스트 인코더로 벡터화해 평균 낸다. 태그는 고정이라
프로세스가 켜져 있는 동안 한 번만 인코딩하고 메모리에 들고 있는다(category_vectors.py와 같다).
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.clients.clip_client import ClipClient
from app.components.tag_vector_file import PrecomputedVectors
from app.components.vector_math import cosine_similarity, normalized_mean

logger = logging.getLogger("muro.axis_vectors")

# app/components/axis_vectors.py 기준 app/axis_tags.yaml
DEFAULT_AXES_PATH = Path(__file__).resolve().parent.parent / "axis_tags.yaml"


@dataclass(frozen=True)
class AxisTagSpec:
    ko: str
    sentences: tuple[str, ...]  # CLIP에 넣을 영어 문장


@dataclass(frozen=True)
class AxisSpec:
    key: str  # 코드에서 쓰는 이름 (kind, subject, vibe, expression)
    ko: str  # 화면·프롬프트에 나가는 이름
    margin: float  # 1등-2등 유사도 차이가 이보다 작으면 불확실
    when: str | None  # 1단계(kind) 결과가 이 값일 때만 본다. None이면 1단계 축이다
    tags: tuple[AxisTagSpec, ...]


@dataclass(frozen=True)
class AxisTagVector:
    ko: str
    vector: list[float]


@dataclass(frozen=True)
class LoadedAxis:
    key: str
    ko: str
    margin: float
    when: str | None
    tags: tuple[AxisTagVector, ...]


def load_axis_specs(path: Path) -> list[AxisSpec]:
    """yaml을 읽고 구조를 검사한다. 벡터화 전 순수 파싱 단계라 테스트하기 쉽다."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_axes = data.get("axes") or []
    if not raw_axes:
        raise ValueError(f"{path}에 축이 하나도 없어요.")

    specs = []
    for raw in raw_axes:
        tags = tuple(
            AxisTagSpec(ko=t["ko"], sentences=tuple(t.get("sentences") or ()))
            for t in raw.get("tags") or []
        )
        spec = AxisSpec(
            key=raw["key"],
            ko=raw["ko"],
            margin=float(raw.get("margin", 0.0)),
            when=raw.get("when"),
            tags=tags,
        )
        if not spec.tags:
            raise ValueError(f"축 {spec.key}에 태그가 하나도 없어요.")
        if spec.margin < 0:
            raise ValueError(f"축 {spec.key}의 margin은 0 이상이어야 해요.")
        for tag in spec.tags:
            if not tag.sentences:
                raise ValueError(f"축 {spec.key}의 태그 {tag.ko}에 문장이 없어요.")
        specs.append(spec)

    keys = [s.key for s in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("축 key가 겹쳐요.")
    roots = [s for s in specs if s.when is None]
    if len(roots) != 1:
        raise ValueError("when이 없는 1단계 축은 정확히 하나여야 해요.")
    root_tags = {t.ko for t in roots[0].tags}
    for spec in specs:
        if spec.when is not None and spec.when not in root_tags:
            raise ValueError(f"축 {spec.key}의 when({spec.when})이 1단계 태그에 없어요.")
    return specs


class AxisVectors:
    """축별 태그 벡터 캐시. `load()`는 첫 사용 때 한 번 부른다."""

    def __init__(
        self,
        clip: ClipClient,
        path: Path = DEFAULT_AXES_PATH,
        precomputed: PrecomputedVectors | None = None,
    ):
        self._clip = clip
        self._path = path
        self._precomputed = precomputed
        self._axes: list[LoadedAxis] = []
        # 요청 여러 개가 동시에 처음 부르면 각자 CLIP을 부르지 않고 한 번만 계산하게 한다
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self._axes)

    @property
    def axes(self) -> list[LoadedAxis]:
        return self._axes

    @property
    def root(self) -> LoadedAxis:
        """1단계 축 (사진 종류)."""
        return next(axis for axis in self._axes if axis.when is None)

    def get(self, key: str) -> LoadedAxis:
        for axis in self._axes:
            if axis.key == key:
                return axis
        raise ValueError(f"알 수 없는 축이에요: {key}")

    def children_of(self, root_tag: str) -> list[LoadedAxis]:
        """1단계 결과가 root_tag일 때 이어서 볼 2단계 축들 (yaml 순서)."""
        return [axis for axis in self._axes if axis.when == root_tag]

    async def load(self) -> None:
        async with self._lock:
            if self._axes:  # 기다리는 동안 다른 요청이 이미 끝냈다
                return
            specs = load_axis_specs(self._path)
            self._axes = self._from_precomputed(specs) or await self._encode(specs)

    def _from_precomputed(self, specs: list[AxisSpec]) -> list[LoadedAxis] | None:
        """미리 계산한 벡터가 모든 축·태그에 있고 CLIP 차원과 맞을 때만 쓴다. 아니면 None."""
        if self._precomputed is None:
            return None
        dim = getattr(self._clip, "dim", None)
        axes = []
        for spec in specs:
            by_tag = self._precomputed.axes.get(spec.key, {})
            vectors = [by_tag.get(tag.ko) for tag in spec.tags]
            if any(v is None or (dim is not None and len(v) != dim) for v in vectors):
                return None
            tags = tuple(
                AxisTagVector(ko=tag.ko, vector=vector)
                for tag, vector in zip(spec.tags, vectors, strict=True)  # type: ignore[arg-type]
            )
            axes.append(
                LoadedAxis(key=spec.key, ko=spec.ko, margin=spec.margin, when=spec.when, tags=tags)
            )
        return axes

    async def _encode(self, specs: list[AxisSpec]) -> list[LoadedAxis]:
        sentences = [s for spec in specs for tag in spec.tags for s in tag.sentences]
        logger.warning(
            "미리 계산한 다축 태그 벡터를 쓸 수 없어 CLIP으로 문장 %d개를 인코딩해요 "
            "(몇 분 걸릴 수 있음)",
            len(sentences),
        )
        vectors = iter(await self._clip.encode_text(sentences))
        axes = []
        for spec in specs:
            tags = []
            for tag in spec.tags:
                tag_vectors = [next(vectors) for _ in tag.sentences]
                tags.append(AxisTagVector(ko=tag.ko, vector=normalized_mean(tag_vectors)))
            axes.append(
                LoadedAxis(
                    key=spec.key, ko=spec.ko, margin=spec.margin, when=spec.when, tags=tuple(tags)
                )
            )
        return axes


def pick_tag(image_vector: list[float], axis: LoadedAxis) -> str | None:
    """사진 벡터와 가장 가까운 태그. 1등과 2등 차이가 margin보다 작으면 None(불확실)."""
    scored = sorted(
        ((tag.ko, cosine_similarity(image_vector, tag.vector)) for tag in axis.tags),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if len(scored) >= 2 and scored[0][1] - scored[1][1] < axis.margin:
        return None
    return scored[0][0]
