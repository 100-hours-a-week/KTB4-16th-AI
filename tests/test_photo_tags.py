from app.components.photo_tags import SCENE_KEY, build_photo_tags, split_photo_tags


def test_build_puts_the_first_scene_tag_under_the_scene_key_next_to_the_axes():
    tags = build_photo_tags(["카페"], {"사진 종류": "진짜 사진", "대상": "여러 사람"})

    assert tags == {"장소": "카페", "사진 종류": "진짜 사진", "대상": "여러 사람"}


def test_build_leaves_out_the_scene_key_when_no_scene_tag_is_confident():
    assert build_photo_tags([], {"사진 종류": "그림·짤"}) == {"사진 종류": "그림·짤"}
    assert build_photo_tags([], {}) == {}


def test_split_is_the_reverse_of_build():
    scene, axes = split_photo_tags({"장소": "카페", "사진 종류": "진짜 사진", "분위기": "노을빛"})

    assert scene == ["카페"]
    assert axes == {"사진 종류": "진짜 사진", "분위기": "노을빛"}  # 장소는 다축에 섞이지 않는다


def test_split_without_scene_key_gives_no_scene_tag():
    assert split_photo_tags({"사진 종류": "그림·짤"}) == ([], {"사진 종류": "그림·짤"})
    assert split_photo_tags({}) == ([], {})


def test_scene_key_is_the_documented_one():
    """RECAP이 읽는 키와 기능4가 쓰는 키가 같은 상수라는 것을 못 박아 둔다."""
    assert SCENE_KEY == "장소"
