from app.components.music_mood_tagger import _parse


def test_parse_keeps_genre_in_vocab():
    parsed = _parse("설명: 잔잔한 곡\n장르: R&B/Soul\n무드: 잔잔한, 그리운")
    assert parsed.genre == "R&B/Soul"
    assert parsed.moods == ("잔잔한", "그리운")


def test_parse_drops_genre_outside_vocab():
    # 예전 목록의 "시티팝"은 13개 장르에 없어서 비운다
    assert _parse("설명: 청량한 곡\n장르: 시티팝\n무드: 설레는").genre == ""
