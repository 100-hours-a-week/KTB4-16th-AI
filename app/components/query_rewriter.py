"""태그·컨텍스트 → Spotify 검색어 리라이팅. 기능1·3이 공유한다.

실측으로 확인한 두 가지가 이 프롬프트의 근거다.
- 시적인 문장("노을 지는 바다가 떠오르는 시티팝")으로 검색하면 문자열 매칭이라
  거의 다 빗나간다. 반면 짧은 장르 단어("시티팝", "발라드")는 실제로 그
  장르 곡이 검색됐다 — Spotify 검색은 의미 이해가 아니라 문자열 매칭이다.
- 그래서 LLM에게 "장르/무드 단어 목록 중에서 골라 압축하라"고 강하게
  제약한다. 자유 작문을 허용하면 다시 시적인 문장으로 돌아간다.
"""

from app.clients.llm_client import LLMClient
from app.exceptions import LLMError

PROMPT_VERSION = "query-rewrite-v1"

# Spotify 앱 자체 무드/장르 카테고리 기준 — 검색 시 실제로 그 장르 곡이
# 걸리는 걸 확인한 단어들. 새 단어를 추가하려면 먼저 검색 결과로 검증할 것.
GENRE_VOCAB = [
    "발라드",
    "인디",
    "재즈",
    "시티팝",
    "어쿠스틱",
    "팝",
    "알앨비",
    "힙합",
    "로파이",
    "댄스",
    "클래식",
    "락",
    "트로트",
    # 아래 6개는 Spotify 공식 장르/무드 필터(2021년 발표)에 실제로 쓰이는 카테고리명.
    # https://newsroom.spotify.com/2021-02-25/... 참고
    "칠",
    "트랩",
    "일렉트로닉",
    "포크",
    "몽환적인",
    "소울풀한",
]

SYSTEM_PROMPT = f"""너는 Spotify 검색어를 만드는 도우미다. Spotify 검색은 의미를 이해하지
못하고 제목·아티스트명 문자열만 매칭하므로, 시적인 문장이 아니라 실제 검색에
쓰일 짧은 키워드를 만들어야 한다.

규칙:
- 아래 장르/무드 단어 중 반드시 1개를 포함한다: {", ".join(GENRE_VOCAB)}
- 그 앞에 분위기를 나타내는 수식어를 최대 1개까지만 붙인다 (예: "잔잔한", "신나는").
- 전체 1~3단어로 만든다. 완전한 문장(주어+서술어)을 쓰지 않는다.
- 좋은 예: "노을 시티팝", "잔잔한 발라드", "밤 재즈"
- 나쁜 예: "노을 지는 바다가 떠오르는 시티팝" (문장), "석양 따뜻한색" (태그 직역)
- 검색어 한 줄만 출력한다. 설명·따옴표를 붙이지 않는다."""


class QueryRewriter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def rewrite_from_tags(self, tags: list[str]) -> str:
        """기능3 — CLIP 무드 태그(한국어) → 검색어."""
        if not tags:
            raise LLMError("태그가 없으면 검색어를 만들 수 없어요.")
        prompt = f"사진에서 뽑은 분위기 태그: {', '.join(tags)}"
        query = await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=64)
        return _clean(query)


def _clean(text: str) -> str:
    return text.strip().strip('"').strip("'").splitlines()[0].strip()
