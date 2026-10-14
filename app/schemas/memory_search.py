"""기능5 기억 검색 — 위키 모델 API 설계 7-1.

응답의 필드 이름과 형식은 백엔드 명세(POST /api/memories/search)와 겹치는 것은 백엔드 쪽에 맞췄다
(answer, records, recordId, musicTrackId, title, artistName, placeId, placeName, date).
백엔드는 AI 응답을 거의 그대로 records로 옮기면 된다.
"""

from pydantic import AliasChoices, Field, field_validator

from app.schemas.common import CamelModel


class MemorySearchRequest(CamelModel):
    # 백엔드 userId는 숫자(Long). 숫자 문자열("35")도 받아서 숫자로 바꾼다.
    user_id: int = Field(gt=0)
    # 백엔드 프론트 계약의 요청 필드는 query다. 예전 이름 queryText도 계속 받는다.
    query_text: str = Field(min_length=1, validation_alias=AliasChoices("query", "queryText"))
    # 프론트는 앞 5건만 채팅에 보이고 "전체보기"(임시 창)에서도 최대 50건까지만 보여주므로,
    # 조건에 맞는 자물쇠를 최대 50건 돌려준다. 백엔드가 limit을 보내면 그 값(1~50)을 따른다.
    limit: int = Field(default=50, ge=1, le=50)

    @field_validator("query_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length는 공백만 있는 질문을 못 거른다 — 임베딩이 빈 텍스트를 거부해 500이 된다
        stripped = value.strip()
        if not stripped:
            raise ValueError("질문이 비어 있어요.")
        return stripped


class MemoryRecordItem(CamelModel):
    """찾은 자물쇠 하나. 백엔드 records[]와 이름이 겹치는 필드는 백엔드 이름을 쓴다."""

    record_id: int  # 공통 규칙: recordId는 숫자(Long)
    music_track_id: int
    title: str
    artist_name: str
    # 자물쇠가 한 번도 없던 곳이면 백엔드 places에 행이 없어 비어 온다
    place_id: int | None = None
    # 백엔드 places.legal_dong_name (예: 성수동1가). 값이 없으면 null
    place_name: str | None = None
    date: str  # 기록한 날짜, "2026.03.14" 형식 (한국시간)
    # 기록 당시 날씨 — 카드의 날씨 아이콘용. CLEAR·CLOUDY·OVERCAST·RAIN·SNOW·RAIN_SNOW·SHOWER 중
    # 하나이고, 백엔드가 날씨를 못 구한 기록은 null (백엔드 명세 records에 추가가 필요하다)
    weather_condition: str | None = None
    # 카드 썸네일(앨범 이미지)용. 백엔드 명세 records에 추가하면 그대로 쓸 수 있다
    album_image_url: str  # DB에 없으면 빈 문자열
    # 아래는 AI 전용 값이다 — 백엔드가 안 써도 된다
    external_track_id: str  # Spotify 곡 ID
    external_url: str  # DB에 없으면 빈 문자열
    similarity_score: float  # 0~1, 순서를 정하는 점수


class SuggestedRecordItem(MemoryRecordItem):
    """모든 조건에는 안 맞고 일부만 맞는 자물쇠. 카드 필드는 records와 같다."""

    # 이 자물쇠가 맞은 조건의 이름 (예: ["서울숲 근처"]). 프론트가 "OO만 맞아요"로 보여준다
    matched_conditions: list[str]


class MemorySearchResponse(CamelModel):
    answer: str
    records: list[MemoryRecordItem]
    # 일부만 맞는 자물쇠 (최대 3건, 조건에 모두 맞는 자물쇠가 5건보다 적을 때만). 없으면 빈 목록.
    # 프론트가 "이런 자물쇠를 찾으시는 건 아닌가요?" 토글을 누르면 카드로 보여준다.
    suggested_records: list[SuggestedRecordItem] = []
