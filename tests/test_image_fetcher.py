import base64

import httpx
import pytest

from app.clients.image_fetcher import MAX_BYTES, ImageFetcher, ImageFetchError


def _fetcher(handler) -> ImageFetcher:
    return ImageFetcher(transport=httpx.MockTransport(handler))


def _ok(content: bytes = b"\xff\xd8jpeg", content_type: str = "image/jpeg"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"content-type": content_type})

    return handler


async def test_returns_a_data_url_with_the_content_type():
    url = await _fetcher(_ok(b"abc", "image/png")).fetch_data_url("https://gcs.example/a.png?sig=1")

    assert url == "data:image/png;base64," + base64.b64encode(b"abc").decode()


async def test_content_type_parameters_are_ignored():
    url = await _fetcher(_ok(b"abc", "image/jpeg; charset=binary")).fetch_data_url("https://x/a")

    assert url.startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "url", ["ftp://x/a.jpg", "file:///etc/passwd", "gs://bucket/a.jpg", "a.jpg"]
)
async def test_only_http_urls_are_accepted(url):
    with pytest.raises(ImageFetchError):
        await _fetcher(_ok()).fetch_data_url(url)


async def test_non_image_content_type_is_rejected():
    with pytest.raises(ImageFetchError, match="형식"):
        await _fetcher(_ok(b"<html>", "text/html")).fetch_data_url("https://x/a")


async def test_too_large_and_empty_photos_are_rejected():
    with pytest.raises(ImageFetchError, match="커요"):
        await _fetcher(_ok(b"x" * (MAX_BYTES + 1))).fetch_data_url("https://x/a")
    with pytest.raises(ImageFetchError, match="비어"):
        await _fetcher(_ok(b"")).fetch_data_url("https://x/a")


async def test_http_error_redirect_and_network_failure_are_fetch_errors():
    def not_found(request):
        return httpx.Response(404)

    def redirect(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    def down(request):
        raise httpx.ConnectError("refused")

    for handler in (not_found, redirect, down):
        with pytest.raises(ImageFetchError):
            await _fetcher(handler).fetch_data_url("https://x/a")
