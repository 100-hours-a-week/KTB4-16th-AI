"""사진 서명 — 저장소 선택(gcs 기본·s3), 키 형태 정규화, 기본값에서 기존 동작 유지."""

from contextlib import asynccontextmanager
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from app.clients.photo_signer import (
    GcsPhotoSigner,
    PassThroughPhotoSigner,
    S3PhotoSigner,
    build_photo_signer,
)
from app.config import Settings

BUCKET = "mulo-v3-prod-840255883218-images"
REGION = "ap-northeast-2"
PHOTO_ENV = ("PHOTO_STORAGE_PROVIDER", "GCS_PHOTO_BUCKET", "S3_PHOTO_BUCKET", "AWS_REGION")


@pytest.fixture
def fake_aws(monkeypatch):
    """가짜 자격 증명. 사전 서명은 계산만 하므로 실제 AWS를 부르지 않는다."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKEFAKEFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _settings(monkeypatch, **values) -> Settings:
    for name in PHOTO_ENV:
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, **values)


def _signer_from(s: Settings):
    return build_photo_signer(
        s.photo_storage_provider,
        gcs_bucket=s.gcs_photo_bucket,
        s3_bucket=s.s3_photo_bucket,
        region=s.aws_region,
        minutes=s.photo_signed_url_minutes,
    )


# --- 키 형태별 정규화 ---


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ("uploads/7/a.jpg", "uploads/7/a.jpg"),
        ("gs://old-gcs-bucket/uploads/7/a.jpg", "uploads/7/a.jpg"),
        (
            "https://storage.googleapis.com/old-gcs-bucket/uploads/7/a.jpg?X-Goog-Signature=1",
            "uploads/7/a.jpg",
        ),
        (f"s3://{BUCKET}/uploads/7/a.jpg", "uploads/7/a.jpg"),
        (
            f"https://{BUCKET}.s3.{REGION}.amazonaws.com/uploads/7/a.jpg?X-Amz-Signature=1",
            "uploads/7/a.jpg",
        ),
    ],
)
def test_s3_signer_normalizes_key_formats(value, key):
    assert S3PhotoSigner(BUCKET, REGION, 120)._object_key(value) == key


@pytest.mark.parametrize(
    "url",
    [
        "https://cdn.example.com/photos/a.jpg",
        f"https://other-bucket.s3.{REGION}.amazonaws.com/uploads/7/a.jpg",
    ],
)
def test_s3_signer_returns_other_https_urls_as_is(monkeypatch, url):
    s = S3PhotoSigner(BUCKET, REGION, 120)
    monkeypatch.setattr(s, "_s3", lambda: pytest.fail("다른 곳 URL에는 서명하지 않는다"))
    assert s.sign(url) == url


# --- s3일 때 사전 서명 URL ---


def test_s3_signer_makes_presigned_get_url(fake_aws):
    url = S3PhotoSigner(BUCKET, REGION, 120).sign("uploads/7/a.jpg")

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == f"{BUCKET}.s3.{REGION}.amazonaws.com"
    assert parsed.path == "/uploads/7/a.jpg"
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Expires"] == ["7200"]  # photo_signed_url_minutes 120분
    assert f"/{REGION}/s3/aws4_request" in query["X-Amz-Credential"][0]
    assert query["X-Amz-Signature"][0]


def test_s3_signer_signs_the_normalized_key(fake_aws):
    url = S3PhotoSigner(BUCKET, REGION, 120).sign("gs://old-gcs-bucket/uploads/7/a.jpg")
    assert urlparse(url).path == "/uploads/7/a.jpg"


def test_s3_settings_build_s3_signer(monkeypatch, fake_aws):
    s = _settings(
        monkeypatch,
        photo_storage_provider="s3",
        gcs_photo_bucket="mulo-photos",
        s3_photo_bucket=BUCKET,
        aws_region=REGION,
    )
    signer = _signer_from(s)
    assert isinstance(signer, S3PhotoSigner)
    assert urlparse(signer.sign("uploads/7/a.jpg")).netloc == f"{BUCKET}.s3.{REGION}.amazonaws.com"


def test_s3_without_bucket_fails_only_for_keys(monkeypatch):
    signer = _signer_from(_settings(monkeypatch, photo_storage_provider="s3"))
    assert isinstance(signer, PassThroughPhotoSigner)
    assert signer.sign("https://cdn.example.com/a.jpg") == "https://cdn.example.com/a.jpg"
    with pytest.raises(ValueError, match="S3_PHOTO_BUCKET이 설정되지 않았어요"):
        signer.sign("uploads/7/a.jpg")


# --- gcs 기본값에서 기존 동작 유지 ---


def test_photo_storage_defaults_keep_gcs(monkeypatch):
    s = _settings(monkeypatch)
    assert s.photo_storage_provider == "gcs"
    assert s.s3_photo_bucket == ""
    assert s.aws_region == ""


def test_gcs_default_builds_the_same_gcs_signer(monkeypatch):
    signer = _signer_from(
        _settings(monkeypatch, gcs_photo_bucket="mulo-photos", s3_photo_bucket=BUCKET)
    )
    assert isinstance(signer, GcsPhotoSigner)
    assert signer._bucket == "mulo-photos"
    assert signer._ttl == timedelta(minutes=120)
    assert signer._object_name("gs://mulo-photos/records/1.jpg") == "records/1.jpg"
    assert (
        signer._object_name("https://storage.googleapis.com/mulo-photos/records/1.jpg?X=1")
        == "records/1.jpg"
    )
    assert signer._object_name("records/1.jpg") == "records/1.jpg"
    # 다른 버킷의 gs:// 값은 예전처럼 건드리지 않는다
    assert signer._object_name("gs://other/records/1.jpg") == "gs://other/records/1.jpg"
    # 다른 곳 https URL은 서명 없이 그대로 (google 라이브러리를 부르지 않는다)
    assert signer.sign("https://cdn.example.com/a.jpg") == "https://cdn.example.com/a.jpg"


def test_gcs_default_without_bucket_behaves_as_before(monkeypatch):
    signer = _signer_from(_settings(monkeypatch, s3_photo_bucket=BUCKET))
    assert isinstance(signer, PassThroughPhotoSigner)
    assert signer.sign("https://cdn.example.com/a.jpg") == "https://cdn.example.com/a.jpg"
    with pytest.raises(ValueError) as e:
        signer.sign("uploads/7/a.jpg")
    assert str(e.value) == "사진 키가 URL이 아닌데 GCS_PHOTO_BUCKET이 설정되지 않았어요."


# --- 동기화 워커가 설정 값을 그대로 넘기는지 ---


async def test_record_sync_passes_photo_settings_to_signer(monkeypatch):
    from app.workers import record_sync_worker as w

    s = _settings(
        monkeypatch,
        backend_mysql_url="mysql+aiomysql://u:p@localhost/db",
        photo_storage_provider="s3",
        gcs_photo_bucket="mulo-photos",
        s3_photo_bucket=BUCKET,
        aws_region=REGION,
    )

    @asynccontextmanager
    async def fake_session():
        yield object()

    used = {}

    async def fake_sync(reader, cursors, jobs, store, signer, batch_size, **kwargs):
        used["signer"] = signer

    monkeypatch.setattr(w, "get_settings", lambda: s)
    monkeypatch.setattr(w.mysql, "get_sessionmaker", lambda: fake_session)
    monkeypatch.setattr(w, "get_sessionmaker", lambda: fake_session)
    monkeypatch.setattr(w, "sync", fake_sync)

    await w.handle({})

    signer = used["signer"]
    assert isinstance(signer, S3PhotoSigner)
    assert signer._bucket == BUCKET
    assert signer._region == REGION
    assert signer._expires_seconds == 120 * 60
