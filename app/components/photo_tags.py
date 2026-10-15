"""저장하는 사진 태그(record_embeddings.photo_tags)의 형식.

기능4가 자물쇠를 저장할 때 만들고, 기능2 RECAP이 읽는다. 두 쪽이 같은 규칙을 쓰도록 여기 둔다.

형식: 축 이름 → 태그. 장소(mood_tags.yaml) 태그는 키 "장소", 다축 태그는 축 이름
(사진 종류·대상·분위기·표정·감정)을 키로 쓴다. 자신 있게 붙일 태그가 없는 축은 키가 없다.
"""

SCENE_KEY = "장소"


def build_photo_tags(scene: list[str], axes: dict[str, str]) -> dict[str, str]:
    """분류 결과(장소 태그 목록, 다축 태그)를 저장 형식으로 만든다."""
    return {**({SCENE_KEY: scene[0]} if scene else {}), **axes}


def split_photo_tags(tags: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """저장 형식을 (장소 태그 목록, 다축 태그)로 되돌린다. build_photo_tags의 반대다."""
    scene = [tags[SCENE_KEY]] if SCENE_KEY in tags else []
    axes = {name: tag for name, tag in tags.items() if name != SCENE_KEY}
    return scene, axes
