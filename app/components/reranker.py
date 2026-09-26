"""후보곡 재랭킹 — 기능3(사진 무드) 사용, 기능1(재랭킹 점수 가중합)이 이어서 재사용.

Spotify 검색은 문자열 매칭이라 검색 결과가 실제로 분위기와 안 맞을 수 있다
(실측: "노을 지는 바다가 떠오르는 시티팝"으로 검색했더니 시티팝이 하나도
없었음). 그래서 지금은 검색 대신 LLM(song_curator)이 실제 곡을 지목하고,
그 곡들 중에서 사진 상황과 제일 잘 맞는 순서로 다시 줄세운다.

**2026-09-26: 점수 방식을 임베딩 코사인 유사도에서 태그 겹침으로 바꿨다.**
이전엔 사진 태그 텍스트와 곡의 LLM 생성 무드 문장을 임베딩으로 비교했는데,
실측 결과 "신나는 콘서트 파티" 사진에 조용한 발라드가 1등으로 나오는 등
자유 문장 임베딩이 에너지·템포 같은 축을 신뢰성 있게 구분 못 했다(문장
형태를 맞춰봐도 개선 안 됨 — 임베딩 모델 자체의 한계로 판단). GENRE_VOCAB/
MOOD_VOCAB이라는 정해진 태그끼리 겹치는 개수로 점수를 매기니 실측에서
정확히 개선됐다(신나는 곡이 상위로, 조용한 곡이 하위로).

이 방식은 song_curator가 곡을 추천하는 바로 그 호출에서 각 곡의 장르·무드
태그까지 같이 받아오므로(추가 호출 없음), 임베딩·트랙무드 캐시·백그라운드
백필이 전부 필요 없어졌다 — 그래서 이 컴포넌트는 동기 함수이고 DB·외부
API 의존이 없다. "캐시 있는 곡이 유리해지는" 편향 문제도 이 캐시 의존성
자체가 없어지면서 같이 해소된다.
"""

from dataclasses import dataclass

# 장르 일치 : 무드 겹침 : 인기도 가중치. 인기도는 현재 Spotify가 필드를 없애서
# 항상 0이라 사실상 죽은 항목이지만, 나중에 다른 소스로 복구할 수 있어 남겨둔다.
GENRE_WEIGHT = 0.35
MOOD_WEIGHT = 0.35
POPULARITY_WEIGHT = 0.3


@dataclass(frozen=True)
class TrackCandidate:
    external_track_id: str
    title: str
    artist_name: str
    spotify_uri: str
    genre: str
    moods: tuple[str, ...]
    album_art_url: str | None = None
    popularity: int = 0  # Spotify 제공값 0~100, 못 받으면 0(가장 불리하게)


class Reranker:
    def rerank(
        self,
        situation_genre: str,
        situation_moods: tuple[str, ...],
        candidates: list[TrackCandidate],
    ) -> list[TrackCandidate]:
        if not candidates:
            return []
        return sorted(
            candidates,
            key=lambda c: self._score(situation_genre, situation_moods, c),
            reverse=True,
        )

    def _score(
        self, situation_genre: str, situation_moods: tuple[str, ...], c: TrackCandidate
    ) -> float:
        genre_score = 1.0 if situation_genre and situation_genre == c.genre else 0.0
        mood_overlap = len(set(situation_moods) & set(c.moods))
        mood_score = mood_overlap / len(situation_moods) if situation_moods else 0.0
        popularity_score = c.popularity / 100
        return (
            GENRE_WEIGHT * genre_score
            + MOOD_WEIGHT * mood_score
            + POPULARITY_WEIGHT * popularity_score
        )
