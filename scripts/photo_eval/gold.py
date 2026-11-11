"""사진 평가용 정답 정보 — 속성 어휘와 정답 파일 읽기.

속성은 서비스가 이미 쓰는 어휘를 그대로 쓴다. 장소는 app/mood_tags.yaml, 사진 종류·대상·분위기·
표정·감정은 app/axis_tags.yaml, 노래 분위기는 app/components/music_tags.py의 MOOD_VOCAB이다.

정답 파일(data/photo_eval/gold_labels.json)의 모양:
    {"labels": {"ph-001": {"split": "tune", "confirmed": true,
                           "장소": ["노을"], "사진 종류": ["진짜 사진"], ...}}}
Claude가 쓴 초안은 gold_labels_draft.json에 있고, 사람이 확인해 고친 것이 gold_labels.json이다.
"""

import json
import runpy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/photo_eval"
DRAFT_PATH = DATA / "gold_labels_draft.json"
GOLD_PATH = DATA / "gold_labels.json"

# 속성 → 한 사진에 고를 수 있는 최대 개수
LIMITS = {"장소": 2, "사진 종류": 1, "대상": 1, "분위기": 2, "표정·감정": 1, "노래 분위기": 3}
ATTRS = list(LIMITS)


def load_vocab() -> dict[str, list[str]]:
    scene = yaml.safe_load((ROOT / "app/mood_tags.yaml").read_text(encoding="utf-8"))["tags"]
    axes = yaml.safe_load((ROOT / "app/axis_tags.yaml").read_text(encoding="utf-8"))["axes"]
    vocab = {"장소": [tag["ko"] for tag in scene]}
    for axis in axes:
        vocab[axis["ko"]] = [tag["ko"] for tag in axis["tags"]]
    vocab["노래 분위기"] = list(
        runpy.run_path(str(ROOT / "app/components/music_tags.py"))["MOOD_VOCAB"]
    )
    missing = [attr for attr in ATTRS if attr not in vocab]
    if missing:
        raise ValueError(f"어휘를 찾지 못한 속성: {missing}")
    return {attr: vocab[attr] for attr in ATTRS}


def load_gold(path: Path) -> dict[str, dict]:
    """정답 파일의 labels를 읽는다. 어휘 밖 값이 있으면 오류로 알려 준다."""
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    vocab = load_vocab()
    for pid, label in labels.items():
        for attr in ATTRS:
            unknown = [v for v in label.get(attr, []) if v not in vocab[attr]]
            if unknown:
                raise ValueError(f"{path.name}의 {pid} {attr}에 어휘 밖 값이 있어요: {unknown}")
    return labels
