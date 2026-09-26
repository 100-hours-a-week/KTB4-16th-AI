"""사진 무드 태그 → 실제 유명 가수의 구체적인 곡 목록 + 장르·무드 태그. 기능3이 쓴다.

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
"""

from dataclasses import dataclass

from app.clients.llm_client import LLMClient
from app.components.music_tags import GENRE_VOCAB, MOOD_VOCAB
from app.exceptions import LLMError

PROMPT_VERSION = "song-curate-v2"

# Spotify 확인 단계에서 일부가 탈락하므로 최종 3곡보다 넉넉히 받는다.
CANDIDATE_COUNT = 8

SYSTEM_PROMPT = f"""너는 사진 분위기에 어울리는 음악을 추천하는 도우미다.

먼저 이 사진 상황 자체에 어울리는 장르·무드를 아래 목록에서만 골라 표시한다:
장르 후보: {", ".join(GENRE_VOCAB)}
무드 후보: {", ".join(MOOD_VOCAB)}

규칙:
- 실제로 발매됐고 대중적으로 알려진 가수의 곡만 추천한다.
- 배경음악·플레이리스트용으로 제작된 곡, 커버·연주 편곡 계정의 곡은 추천하지 않는다.
- 정확히 {CANDIDATE_COUNT}개를 추천한다.
- 각 곡에도 위 목록에서만 장르 1개, 무드 1~2개를 붙인다.
- 존재가 불확실한 곡은 추천하지 않는다.

형식(정확히 이대로, 다른 말 붙이지 않는다):
상황: 장르=X | 무드=Y,Z
곡:
아티스트 - 곡명 | 장르=X | 무드=Y,Z
(위 줄을 {CANDIDATE_COUNT}번 반복)"""


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
    situation_genre: str
    situation_moods: tuple[str, ...]
    songs: list[SongCandidate]


class SongCurator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def curate_from_tags(self, tags: list[str]) -> CurationResult:
        if not tags:
            raise LLMError("태그가 없으면 곡을 추천할 수 없어요.")
        prompt = f"사진에서 뽑은 분위기 태그: {', '.join(tags)}"
        response = await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=768)
        return _parse(response)


def _parse_tags_field(text: str) -> tuple[str, tuple[str, ...]]:
    genre = ""
    moods: tuple[str, ...] = ()
    for part in text.split("|"):
        part = part.strip()
        if part.startswith("장르="):
            genre = part.removeprefix("장르=").strip()
        elif part.startswith("무드="):
            moods = tuple(m.strip() for m in part.removeprefix("무드=").split(",") if m.strip())
    return genre, moods


def _parse(text: str) -> CurationResult:
    """LLM이 형식을 어겨도 파싱되는 줄만 살린다 — 한 줄 깨졌다고 전체를 버리지 않는다."""
    lines = [raw.strip() for raw in text.strip().splitlines() if raw.strip()]

    situation_genre, situation_moods = "", ()
    songs: list[SongCandidate] = []
    for line in lines:
        if line.startswith("상황:"):
            situation_genre, situation_moods = _parse_tags_field(line.removeprefix("상황:"))
            continue
        if line == "곡:" or " - " not in line:
            continue

        name_part, _, rest = line.partition("|")
        artist, sep, title = name_part.strip().lstrip("-").strip().partition(" - ")
        if not sep:
            continue
        genre, moods = _parse_tags_field("|" + rest)
        artist, title = artist.strip(), title.strip()
        if artist and title:
            songs.append(SongCandidate(artist=artist, title=title, genre=genre, moods=moods))

    return CurationResult(
        situation_genre=situation_genre, situation_moods=situation_moods, songs=songs
    )
