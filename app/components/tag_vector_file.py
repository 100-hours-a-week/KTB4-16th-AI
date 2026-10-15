"""미리 계산해 둔 태그 벡터 파일(app/tag_vectors.json).

장소 태그(mood_tags.yaml)와 다축 태그(axis_tags.yaml)의 문장을 CLIP으로 벡터로 바꾸는 일은
Replicate를 문장마다 한 번씩 순서대로 불러서 몇 분이 걸린다(실측 583초). 서버가 켜진 뒤 첫
요청이 이걸 하면 백엔드가 기다리지 못해 502가 나므로, 미리 계산해서 이 파일에 저장해 두고
서버는 읽기만 한다.

파일은 scripts/build_tag_vectors.py가 만든다. 태그 정의나 CLIP 모델을 바꾸면
version(tag_version.py)이 달라져서 서버가 이 파일을 쓰지 않으니, 그때는 스크립트를 다시
실행해 파일을 새로 만들어야 한다.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# app/components/tag_vector_file.py 기준 app/tag_vectors.json
DEFAULT_VECTORS_PATH = Path(__file__).resolve().parent.parent / "tag_vectors.json"

# 벡터 값을 소수 6자리까지만 저장한다 — 코사인 유사도에는 영향이 없고 파일이 작아진다
DECIMALS = 6


@dataclass(frozen=True)
class PrecomputedVectors:
    version: str  # tag_version.compute_tag_version 값: CLIP 모델 + 태그 정의
    model: str  # 이 벡터를 만든 CLIP 모델 이름
    scene: dict[str, list[float]]  # 장소 태그의 clip 라벨 → 벡터
    axes: dict[str, dict[str, list[float]]]  # 축 key → 태그 이름 → 벡터


def read_vectors_file(path: Path = DEFAULT_VECTORS_PATH) -> PrecomputedVectors | None:
    """파일을 읽는다. 없거나 형식이 깨졌으면 None (그 경우 서버는 CLIP으로 직접 계산한다)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PrecomputedVectors(
            version=str(data["version"]),
            model=str(data["model"]),
            scene={label: _floats(v) for label, v in data["scene"].items()},
            axes={
                axis: {tag: _floats(v) for tag, v in tags.items()}
                for axis, tags in data["axes"].items()
            },
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_vectors_file(path: Path, vectors: PrecomputedVectors) -> None:
    payload = {
        "version": vectors.version,
        "model": vectors.model,
        "scene": {label: _rounded(v) for label, v in vectors.scene.items()},
        "axes": {
            axis: {tag: _rounded(v) for tag, v in tags.items()}
            for axis, tags in vectors.axes.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")


def from_loaded(version: str, model: str, categories, axes) -> PrecomputedVectors:
    """이미 CLIP으로 계산한 CategoryVectors·AxisVectors의 벡터를 저장용으로 옮긴다."""
    return PrecomputedVectors(
        version=version,
        model=model,
        scene={tag.clip_label: tag.vector for tag in categories.tags},
        axes={axis.key: {tag.ko: tag.vector for tag in axis.tags} for axis in axes.axes},
    )


def _floats(values) -> list[float]:
    return [float(x) for x in values]


def _rounded(values: list[float]) -> list[float]:
    return [round(x, DECIMALS) for x in values]
