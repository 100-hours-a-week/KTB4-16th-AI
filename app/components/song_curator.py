"""상황 설명 → 실제 유명 가수의 구체적인 곡 목록 + 장르·무드 태그. 기능1·3이 공유한다.

기능3은 사진 CLIP 태그를, 기능1은 장소·날씨·시간대(+ 비슷한 상황에서 들었던 곡)를
상황 설명으로 넘긴다.

장르 키워드로 Spotify를 검색하는 방식(query_rewriter.py)을 이걸로 대체했다. 실측 결과
"카페 어쿠스틱" 같은 장르 키워드로 검색하면, 그 키워드를 제목에 그대로 박아둔
배경음악·컴필레이션 계정의 곡이 상위를 차지한다. 원인은 검색 엔진이 아니라 질의 자체다 —
유명 가수는 곡 제목에 장르 키워드를 안 붙이기 때문에, 장르 키워드로 검색하는 한
어느 플랫폼에서든 무드용으로 제작된 콘텐츠가 걸린다(YouTube에서도 동일하게 재현됨).

그래서 "검색으로 곡을 찾는" 대신 "LLM이 아는 실제 곡을 지목하게" 한다. 곡·가수 지식이
필요한 작업이므로 llm_knowledge를 쓴다. 지목된 곡이 실존하는지는 Spotify 검색으로
확인하고(photo_recommend_service._resolve_on_spotify), 없는 곡은 버린다 — 할루시네이션
방어가 이 확인 단계다.

**2026-09-26: 재랭킹 기준을 임베딩 유사도에서 태그 겹침으로 바꾸면서, 곡 추천과
동시에 사진 상황·곡 각각의 장르·무드 태그도 같은 호출에서 받는다.** 실측(임베딩
방식)에서 "신나는 콘서트 파티" 사진에 조용한 발라드가 1등으로 나오는 문제가 있었는데,
자유 문장 임베딩이 "에너지·템포" 같은 축을 잘 구분 못 해서였다. GENRE_VOCAB/MOOD_VOCAB
같은 정해진 태그로 겹침을 세는 방식이 실측으로 더 정확했다(같은 사진 태그로 테스트 시
신나는 곡이 정확히 상위로, 조용한 곡이 하위로 감). 태그를 별도 호출로 안 받고 이
호출에 얹은 이유는 추가 지연/비용 없이 되기 때문이다.

**상황 장르는 여러 개(1~3), 무드는 2~3개로 받는다.** 상황 장르를 1개로만 받았더니
LLM이 거의 매번 "시티팝"을 찍어놓고 정작 다른 장르 곡을 추천해서, 장르가 딱 일치하는
곡이 거의 없었다(실측: 한강공원 저녁 → 상황 "시티팝", 추천 8곡 중 시티팝 1곡). "맑은
저녁 한강공원"처럼 장르 하나로 안 떨어지는 상황이 많아서 후보 장르를 여러 개로 넓혔다.
"""

from dataclasses import dataclass

from app.clients.llm_client import LLMClient
from app.components.music_tags import GENRE_VOCAB, MOOD_VOCAB
from app.exceptions import LLMError

PROMPT_VERSION = "song-curate-v5"

# Spotify 확인 단계에서 일부가 탈락하므로 최종 3곡보다 넉넉히 받는다.
CANDIDATE_COUNT = 8

SYSTEM_PROMPT = f"""너는 주어진 상황에 어울리는 음악을 추천하는 도우미다.

먼저 이 상황 자체에 어울리는 장르를 1~3개, 무드를 2~3개 아래 목록에서만 고른다:
장르 후보: {", ".join(GENRE_VOCAB)}
무드 후보: {", ".join(MOOD_VOCAB)}

그다음 곡을 추천한다.
규칙:
- 추천하는 곡은 위에서 고른 상황 장르·무드에 맞는 곡으로 고른다.
- 실제로 발매됐고 대중적으로 알려진 가수의 곡만 추천한다.
- 배경음악·플레이리스트용으로 제작된 곡, 커버·연주 편곡 계정의 곡은 추천하지 않는다.
- 정확히 {CANDIDATE_COUNT}개를 추천한다.
- 각 곡에도 위 목록에서만 장르 1개, 무드 1~2개를 붙인다.
- 존재가 불확실한 곡은 추천하지 않는다.
- 곡 제목·가수 이름은 Spotify 표기 그대로 쓴다(영어 제목을 한글로 옮기지 않는다).
- 사용자가 예전에 들었던 곡이 주어지면 그 취향을 참고하되, 그 곡 자체는 다시 추천하지 않는다.

출력 형식(꺾쇠 <> 부분만 실제 값으로 바꿔 쓰고, 다른 말은 붙이지 않는다):
상황: 장르=<장르>,<장르> | 무드=<무드>,<무드>
곡:
<가수> - <곡 제목> | 장르=<장르> | 무드=<무드>,<무드>
(곡 줄을 {CANDIDATE_COUNT}번 쓴다)"""


@dataclass(frozen=True)
class SongCandidate:
    artist: str
    title: str
    genre: str
    moods: tuple[str, ...]

    @property
    def search_query(self) -> str:
        """Spotify 정밀 검색용 질의 — track:/artist: 필드 필터로 제목·가수를 둘 다 고정한다.

        그냥 "제목 가수"로 검색하면 LLM이 지어낸 곡이어도 Spotify가 0건을 안 주고
        비슷한 아무 곡이나 1위로 돌려준다(실측: "정승환 - 노을" → "노라조 - 형",
        "윤딴딴 - Bad Habits" → "Ed Sheeran - Bad Habits"). 필드 필터를 쓰면 없는
        조합은 0건이 나와 할루시네이션이 걸러지고, "크러쉬"/"Crush" 같은 표기
        차이는 Spotify가 알아서 맞춰준다.
        """
        return f"track:{self.title} artist:{self.artist}"


@dataclass(frozen=True)
class CurationResult:
    situation_genres: tuple[str, ...]
    situation_moods: tuple[str, ...]
    songs: list[SongCandidate]


class SongCurator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def curate_from_tags(self, tags: list[str]) -> CurationResult:
        """기능3 — 사진 CLIP 태그로 추천."""
        if not tags:
            raise LLMError("태그가 없으면 곡을 추천할 수 없어요.")
        return await self.curate(f"사진에서 뽑은 분위기 태그: {', '.join(tags)}")

    async def curate(self, situation: str) -> CurationResult:
        """상황 설명 문장으로 추천. 기능1은 장소·날씨·시간대를 여기로 넘긴다."""
        if not situation.strip():
            raise LLMError("상황 설명이 없으면 곡을 추천할 수 없어요.")
        response = await self._llm.complete(system=SYSTEM_PROMPT, prompt=situation, max_tokens=768)
        return _parse(response)


def _pick(values: str, vocab: list[str]) -> tuple[str, ...]:
    """쉼표로 나열된 태그 중 어휘 목록에 있는 것만 남긴다 — 목록 밖 태그는 겹침 계산이 안 된다."""
    picked = [v.strip() for v in values.split(",")]
    return tuple(v for v in picked if v in vocab)


def _parse_tags_field(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    genres: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    for part in text.split("|"):
        part = part.strip()
        if part.startswith("장르="):
            genres = _pick(part.removeprefix("장르="), GENRE_VOCAB)
        elif part.startswith("무드="):
            moods = _pick(part.removeprefix("무드="), MOOD_VOCAB)
    return genres, moods


def _parse(text: str) -> CurationResult:
    """LLM이 형식을 어겨도 파싱되는 줄만 살린다 — 한 줄 깨졌다고 전체를 버리지 않는다."""
    lines = [raw.strip() for raw in text.strip().splitlines() if raw.strip()]

    situation_genres: tuple[str, ...] = ()
    situation_moods: tuple[str, ...] = ()
    songs: list[SongCandidate] = []
    for line in lines:
        if line.startswith("상황:"):
            situation_genres, situation_moods = _parse_tags_field(line.removeprefix("상황:"))
            continue
        # LLM이 형식 예시 줄을 곡처럼 그대로 베끼는 경우가 있었다(실측)
        if line == "곡:" or " - " not in line or "<" in line:
            continue

        name_part, _, rest = line.partition("|")
        artist, sep, title = name_part.strip().lstrip("-").strip().partition(" - ")
        if not sep:
            continue
        genres, moods = _parse_tags_field("|" + rest)
        artist, title = artist.strip(), title.strip()
        if artist and title:
            songs.append(
                SongCandidate(
                    artist=artist, title=title, genre=genres[0] if genres else "", moods=moods
                )
            )

    return CurationResult(
        situation_genres=situation_genres, situation_moods=situation_moods, songs=songs
    )
