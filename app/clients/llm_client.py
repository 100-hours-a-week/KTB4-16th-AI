"""LLM 인터페이스 — 구현체 교체 시 이 계약만 지키면 된다."""

import json
from typing import Protocol

import httpx

from app.exceptions import LLMError


class LLMClient(Protocol):
    model: str

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        json_mode: bool = False,
        image_url: str | None = None,
    ) -> str: ...


class LocalLLMClient:
    """RunPod에서 직접 서빙하는 로컬 LLM(OpenAI 호환 /v1/chat/completions, 현재 Gemma-3-12B).

    url은 /v1/chat/completions까지 포함한 전체 주소다. json_mode를 켜면 서버에 JSON 객체만
    내도록 요청한다(vLLM response_format).
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        temperature: float = 0.7,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model or "local-llm"
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._transport = transport

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        json_mode: bool = False,
        image_url: str | None = None,
    ) -> str:
        """image_url(https 주소나 data: URL)을 주면 모델이 사진도 같이 본다 (Gemma 3은 가능)."""
        if not self._url:
            raise LLMError("로컬 LLM 주소(LOCAL_LLM_URL)가 설정되지 않았어요.")
        user_content: str | list = prompt
        if image_url:
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        body: dict = {
            "model": self.model,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            client_args = {"timeout": self._timeout, "transport": self._transport}
            async with httpx.AsyncClient(**client_args) as client:
                res = await client.post(self._url, json=body, headers=headers)
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"로컬 LLM 오류 (status {e.response.status_code})") from e
        except httpx.HTTPError as e:
            raise LLMError("로컬 LLM 연결 실패") from e
        except (KeyError, IndexError, ValueError, AttributeError) as e:
            raise LLMError("로컬 LLM 응답 형식이 달라요.") from e
        if not text:
            raise LLMError("로컬 LLM 응답이 비어 있어요.")
        return text


class FakeLLMClient:
    model = "fake-llm"

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        json_mode: bool = False,
        image_url: str | None = None,
    ) -> str:
        if json_mode:
            # 장면 문장(scene_curator)용 — 장면 3개가 든 JSON
            scenes = [
                {
                    "scene": f"[fake] 장면 {i}",
                    "description": f"[fake] 잔잔하고 따뜻한 곡이다. 마음 {i}. 이야기 {i}.",
                    "lyric": f"[fake] 오늘은 {i} / 조용히 걸어 / 너를 떠올려 / 괜찮아",
                }
                for i in (1, 2, 3)
            ]
            return json.dumps({"scenes": scenes}, ensure_ascii=False)
        # music_mood_tagger.py의 "설명:/장르:/무드:" 형식에 맞춰둔다 — 이 클라이언트를
        # 쓰는 다른 컴포넌트(song_curator 등)는 이 형식에서 자기가 원하는 줄을 못 찾으면
        # 그냥 빈 결과로 처리하도록 이미 설계돼 있다(할루시네이션 방어와 같은 경로).
        return (
            f"설명: [fake] {prompt[:80]} 에 어울리는 잔잔하고 따뜻한 분위기\n"
            "장르: 발라드\n"
            "무드: 잔잔한,따뜻한"
        )
