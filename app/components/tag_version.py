"""사진 태그 목록의 버전 — 저장된 태그가 어떤 목록으로 분류됐는지 구분한다.

CLIP 모델 이름과 태그 정의(장소 태그, 다축 태그의 문장·margin)로 만든 짧은 해시다. 태그 목록을
바꾸면 값이 달라져서, 저장된 이미지 벡터로 다시 분류해야 하는 행을 찾을 수 있다. yaml 파일의
주석이나 줄바꿈만 고친 경우에는 값이 바뀌지 않는다(읽어 낸 내용으로 만들기 때문).
"""

import hashlib
from pathlib import Path

from app.components.axis_vectors import DEFAULT_AXES_PATH, load_axis_specs
from app.components.category_vectors import DEFAULT_TAGS_PATH, load_tag_labels


def compute_tag_version(
    clip_model: str,
    mood_tags_path: Path = DEFAULT_TAGS_PATH,
    axes_path: Path = DEFAULT_AXES_PATH,
) -> str:
    content = repr((clip_model, load_tag_labels(mood_tags_path), load_axis_specs(axes_path)))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
