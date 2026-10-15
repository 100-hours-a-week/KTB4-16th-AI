from pathlib import Path

from app.components.tag_version import compute_tag_version

FIXTURES = Path(__file__).parent / "fixtures"
SCENE_YAML = FIXTURES / "mood_tags_sample.yaml"
AXIS_YAML = FIXTURES / "axis_tags_sample.yaml"


def _copy_axes(tmp_path: Path, transform) -> Path:
    path = tmp_path / "axes.yaml"
    path.write_text(transform(AXIS_YAML.read_text(encoding="utf-8")), encoding="utf-8")
    return path


def test_same_inputs_give_the_same_version():
    assert compute_tag_version("clip", SCENE_YAML, AXIS_YAML) == compute_tag_version(
        "clip", SCENE_YAML, AXIS_YAML
    )


def test_version_changes_with_the_clip_model():
    assert compute_tag_version("clip-a", SCENE_YAML, AXIS_YAML) != compute_tag_version(
        "clip-b", SCENE_YAML, AXIS_YAML
    )


def test_version_changes_when_a_tag_sentence_changes(tmp_path):
    changed = _copy_axes(tmp_path, lambda text: text.replace("cartoon", "a drawing"))

    assert compute_tag_version("clip", SCENE_YAML, AXIS_YAML) != compute_tag_version(
        "clip", SCENE_YAML, changed
    )


def test_version_changes_when_margin_changes(tmp_path):
    changed = _copy_axes(tmp_path, lambda text: text.replace("margin: 0.1", "margin: 0.2", 1))

    assert compute_tag_version("clip", SCENE_YAML, AXIS_YAML) != compute_tag_version(
        "clip", SCENE_YAML, changed
    )


def test_comment_only_edits_do_not_change_the_version(tmp_path):
    """주석·빈 줄만 고친 경우에는 저장된 태그를 다시 계산할 필요가 없다."""
    changed = _copy_axes(tmp_path, lambda text: "# 주석만 추가\n\n" + text)

    assert compute_tag_version("clip", SCENE_YAML, AXIS_YAML) == compute_tag_version(
        "clip", SCENE_YAML, changed
    )


def test_default_files_give_a_short_hex_version():
    version = compute_tag_version("openai/clip")

    assert len(version) == 12
    int(version, 16)  # 16진수 문자열이어야 한다
