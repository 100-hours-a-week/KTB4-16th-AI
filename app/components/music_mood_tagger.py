"""곡 메타데이터 → 무드 설명 텍스트 + 장르·무드 태그.

Spotify가 음원을 주지 않아 소리를 직접 임베딩할 수 없으므로,
LLM이 제목·아티스트를 보고 무드를 글로 설명하고 그 텍스트를 임베딩한다.
이 자유 문장+임베딩은 기능5의 자유 검색("그때 그 잔잔한 노래")처럼 사용자가
아무 말이나 쓰는 상황에 쓴다.

**2026-09-26**: 여기에 장르·무드 태그(song_curator.py와 같은 GENRE_VOCAB/
MOOD_VOCAB)를 추가했다. 기능3 재랭킹이 자유 문장 임베딩 대신 이 태그 겹침으로
바뀌었기 때문에(자유 문장 임베딩이 "신나는지 잔잔한지" 구분을 신뢰성 있게
못 해서), song_curator를 거치지 않고 기능4로 직접 저장되는 곡(사용자가 직접
고른 곡 등)도 같은 태그 체계를 갖도록 통일했다.
"""

from dataclasses import dataclass

from app.clients.llm_client import LLMClient
from app.components.music_tags import GENRE_VOCAB, MOOD_VOCAB

PROMPT_VERSION = "music-mood-v3"

SYSTEM_PROMPT = f"""너는 음악 큐레이터다. 주어진 곡의 분위기를 한국어로 설명한다.

형식(정확히 이대로, 다른 말 붙이지 않는다):
설명: <1~2문장으로 곡의 정서·템포·어울리는 상황 묘사>
장르: <아래 목록 중 1개>
무드: <아래 목록 중 1~2개, 쉼표로>

장르 후보: {", ".join(GENRE_VOCAB)}
무드 후보: {", ".join(MOOD_VOCAB)}

곡을 확실히 모르면 아티스트의 대표적인 스타일을 근거로 추정한다."""


@dataclass(frozen=True)
class MoodDescription:
    text: str
    genre: str
    moods: tuple[str, ...]


class MusicMoodTagger:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def describe(self, title: str, artist: str) -> MoodDescription:
        prompt = f"제목: {title}\n아티스트: {artist}"
        response = await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=512)
        return _parse(response)


def _parse(text: str) -> MoodDescription:
    description, genre, moods = "", "", ()
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("설명:"):
            description = line.removeprefix("설명:").strip()
        elif line.startswith("장르:"):
            genre = line.removeprefix("장르:").strip()
        elif line.startswith("무드:"):
            moods = tuple(m.strip() for m in line.removeprefix("무드:").split(",") if m.strip())
    return MoodDescription(text=description, genre=genre, moods=moods)
