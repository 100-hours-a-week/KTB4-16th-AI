from app.components.reranker import Reranker, TrackCandidate


def _candidate(track_id, genre, moods, popularity=0) -> TrackCandidate:
    return TrackCandidate(
        external_track_id=track_id,
        title=f"Song {track_id}",
        artist_name="Artist",
        spotify_uri=f"spotify:track:{track_id}",
        genre=genre,
        moods=moods,
        popularity=popularity,
    )


def test_empty_candidates_short_circuits():
    assert Reranker().rerank("댄스", ("신나는", "화려한"), []) == []


def test_exact_genre_and_mood_match_ranks_first():
    """장르·무드 다 맞는 곡이 하나도 안 맞는 곡보다 위로 온다 — 실측으로 확인한 문제 재현 방지.

    "신나는 콘서트 파티" 상황에 조용한 발라드가 1등으로 나오던 문제(임베딩 유사도 방식)가
    태그 겹침 방식으로 바뀌면서 고쳐졌는지 확인한다.
    """
    party_song = _candidate("upbeat", genre="댄스", moods=("신나는", "화려한"))
    calm_song = _candidate("calm", genre="발라드", moods=("그리운", "따뜻한"))

    ranked = Reranker().rerank("댄스", ("신나는", "화려한"), [calm_song, party_song])

    assert [c.external_track_id for c in ranked] == ["upbeat", "calm"]


def test_partial_mood_overlap_ranks_between_full_and_no_match():
    full_match = _candidate("full", genre="댄스", moods=("신나는", "화려한"))
    partial_match = _candidate("partial", genre="힙합", moods=("신나는", "그리운"))
    no_match = _candidate("none", genre="발라드", moods=("그리운", "따뜻한"))

    ranked = Reranker().rerank("댄스", ("신나는", "화려한"), [no_match, partial_match, full_match])

    assert [c.external_track_id for c in ranked] == ["full", "partial", "none"]


def test_genre_match_alone_beats_no_match_at_all():
    genre_only = _candidate("genre_only", genre="댄스", moods=("그리운",))
    nothing = _candidate("nothing", genre="발라드", moods=("그리운",))

    ranked = Reranker().rerank("댄스", ("신나는",), [nothing, genre_only])

    assert [c.external_track_id for c in ranked] == ["genre_only", "nothing"]


def test_popularity_breaks_ties_between_equal_tag_matches():
    unpopular = _candidate("unpopular", genre="댄스", moods=("신나는",), popularity=0)
    popular = _candidate("popular", genre="댄스", moods=("신나는",), popularity=100)

    ranked = Reranker().rerank("댄스", ("신나는",), [unpopular, popular])

    assert [c.external_track_id for c in ranked] == ["popular", "unpopular"]


def test_no_situation_moods_does_not_crash_and_uses_genre_only():
    song = _candidate("a", genre="댄스", moods=("신나는",))
    ranked = Reranker().rerank("댄스", (), [song])
    assert [c.external_track_id for c in ranked] == ["a"]
