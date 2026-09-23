"""곡 메타데이터 → 무드 설명 텍스트.

Spotify가 음원을 주지 않아 소리를 직접 임베딩할 수 없으므로,
LLM이 제목·아티스트를 보고 무드를 글로 설명하고 그 텍스트를 임베딩한다.
"""

from app.clients.llm_client import LLMClient

PROMPT_VERSION = "music-mood-v2"

SYSTEM_PROMPT = """너는 음악 큐레이터다. 주어진 곡의 분위기를 한국어로 설명한다.
- 1~2문장으로 곡의 정서·템포·어울리는 상황을 묘사한다.
- 마지막 줄에 "태그: " 뒤에 무드 키워드 3~5개를 쉼표로 적는다.
- 곡을 확실히 모르면 아티스트의 대표적인 스타일을 근거로 추정한다.
- 설명과 태그 외의 말은 쓰지 않는다."""


class MusicMoodTagger:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def describe(self, title: str, artist: str) -> str:
        prompt = f"제목: {title}\n아티스트: {artist}"
        return await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=512)
