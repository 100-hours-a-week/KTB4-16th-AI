from app.components.recommend_reason import context_reason, photo_reason, top_genres


def test_top_genres_by_count_then_first_seen_and_unknown_values_dropped():
    genres = ["댄스", "발라드", "발라드", "공통", "R&B/Soul", "댄스", "시티팝"]

    assert top_genres(genres) == ["댄스", "발라드"]
    assert top_genres(genres, limit=3) == ["댄스", "발라드", "R&B/Soul"]
    assert top_genres([]) == []


def test_context_reason_with_situation_genres_and_taste():
    text = context_reason(
        "비 오는 겨울 평일 저녁", ["발라드", "R&B/Soul", "발라드"], ["발라드", "인디음악"]
    )

    assert text == (
        "비 오는 겨울 평일 저녁이라 발라드·R&B/Soul 위주로 골라봤어요. "
        "좋아하시는 장르(발라드·인디음악)도 반영했어요."
    )


def test_context_reason_without_taste_or_genres():
    assert (
        context_reason("맑은 봄 주말 낮", ["댄스"]) == "맑은 봄 주말 낮이라 댄스 위주로 골라봤어요."
    )
    assert (
        context_reason("맑은 봄 주말 낮", []) == "맑은 봄 주말 낮이라 어울리는 곡으로 골라봤어요."
    )
    assert context_reason("", ["댄스"]) == "댄스 위주로 골라봤어요."


def test_taste_list_is_capped_at_three():
    text = context_reason("밤", ["댄스"], ["발라드", "댄스", "POP", "OST"])

    assert "좋아하시는 장르(발라드·댄스·POP)" in text


def test_photo_reason():
    assert photo_reason(["인디음악", "포크/블루스", "인디음악"]) == (
        "사진 분위기에 맞춰 인디음악·포크/블루스 위주로 골라봤어요."
    )
    assert photo_reason([]) == "사진 분위기에 어울리는 곡으로 골라봤어요."


def test_photo_reason_with_a_summary():
    assert photo_reason(["인디음악"], "노을 지는 바닷가") == (
        "노을 지는 바닷가 사진이라 인디음악 위주로 골라봤어요."
    )
    assert (
        photo_reason([], "비 오는 카페 창가")
        == "비 오는 카페 창가 사진이라 어울리는 곡으로 골라봤어요."
    )
