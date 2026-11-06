import json

import httpx
import pytest

from app.clients.llm_client import FakeLLMClient, LocalLLMClient
from app.components.scene_curator import parse_scenes
from app.exceptions import LLMError

URL = "https://gemma.example/v1/chat/completions"


def _client(handler, **kwargs) -> LocalLLMClient:
    return LocalLLMClient(
        URL, "gemma-3-12b", api_key="k", transport=httpx.MockTransport(handler), **kwargs
    )


def _ok(content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


async def test_sends_openai_compatible_request_and_returns_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " 안녕 "}}]})

    text = await _client(handler).complete(system="시스템", prompt="질문", max_tokens=50)

    assert text == "안녕"
    assert seen["url"] == URL
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "gemma-3-12b"
    assert seen["body"]["max_tokens"] == 50
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "시스템"},
        {"role": "user", "content": "질문"},
    ]
    assert "response_format" not in seen["body"]


async def test_json_mode_asks_for_a_json_object():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    await _client(handler).complete(system="s", prompt="p", json_mode=True)

    assert seen["body"]["response_format"] == {"type": "json_object"}


async def test_missing_url_is_an_llm_error():
    with pytest.raises(LLMError):
        await LocalLLMClient("", "gemma").complete(system="s", prompt="p")


async def test_http_status_error_is_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(LLMError, match="503"):
        await _client(handler).complete(system="s", prompt="p")


async def test_connection_error_is_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        await _client(handler).complete(system="s", prompt="p")


async def test_unexpected_body_and_empty_text_are_llm_errors():
    def odd(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": 1})

    with pytest.raises(LLMError):
        await _client(odd).complete(system="s", prompt="p")
    with pytest.raises(LLMError):
        await _client(_ok("   ")).complete(system="s", prompt="p")


async def test_fake_client_json_mode_returns_three_parseable_scenes():
    text = await FakeLLMClient().complete(system="s", prompt="p", json_mode=True)

    assert len(parse_scenes(text)) == 3


async def test_image_url_is_sent_as_an_image_part_next_to_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    await _client(handler).complete(
        system="s", prompt="사진을 봐", image_url="data:image/jpeg;base64,AAAA"
    )

    assert seen["body"]["messages"][1]["content"] == [
        {"type": "text", "text": "사진을 봐"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]


async def test_without_an_image_the_user_message_stays_plain_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    await _client(handler).complete(system="s", prompt="p")

    assert seen["body"]["messages"][1]["content"] == "p"
