"""자물쇠 사진 서명 URL 생성 — 기능4 동기화가 백엔드 DB의 사진 키로 사진 URL을 만든다.

V1에서는 백엔드가 서명 URL을 만들어 보내 줬지만, pull로 바꾸면서 AI가 직접 만든다.
저장소는 PHOTO_STORAGE_PROVIDER로 고른다 (기본 gcs).

- gcs: GCP VM의 서비스 계정으로 서명하므로 키 파일이 필요 없고, 대신 그 서비스 계정에
  '서비스 계정 토큰 생성자' 역할이 필요하다 (클라우드 팀 설정).
- s3: 자격 증명을 코드에 두지 않고 boto3 기본 체인(노드 IAM 역할 등)으로 서명한다.
  AI 노드에 버킷 읽기 권한이 필요하다 (인프라 팀 설정).
"""

import re
from datetime import timedelta
from typing import Literal, Protocol


class PhotoSigner(Protocol):
    def sign(self, key: str) -> str: ...


class GcsPhotoSigner:
    def __init__(self, bucket: str, minutes: int):
        self._bucket = bucket
        self._ttl = timedelta(minutes=minutes)

    def sign(self, key: str) -> str:
        key = self._object_name(key)
        if key.startswith("https://"):
            return key  # 다른 곳에 있는 공개 URL
        import google.auth
        from google.auth.transport.requests import Request
        from google.cloud import storage

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        blob = storage.Client(credentials=credentials).bucket(self._bucket).blob(key)
        return blob.generate_signed_url(
            version="v4",
            expiration=self._ttl,
            method="GET",
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )

    def _object_name(self, value: str) -> str:
        """백엔드가 GCS 경로를 어떤 형태로 저장했든 버킷 안 파일 이름으로 바꾼다.
        gs://버킷/키, https://storage.googleapis.com/버킷/키, 키 그대로 모두 받는다."""
        for prefix in (f"gs://{self._bucket}/", f"https://storage.googleapis.com/{self._bucket}/"):
            if value.startswith(prefix):
                return value[len(prefix) :].split("?", 1)[0]
        return value


class S3PhotoSigner:
    def __init__(self, bucket: str, region: str, minutes: int):
        self._bucket = bucket
        self._region = region
        self._expires_seconds = int(timedelta(minutes=minutes).total_seconds())
        self._bucket_url = re.compile(
            rf"^https://{re.escape(bucket)}\.s3\.[a-z0-9-]+\.amazonaws\.com/"
        )
        self._client = None

    def sign(self, key: str) -> str:
        key = self._object_key(key)
        if key.startswith("https://"):
            return key  # 다른 곳에 있는 공개 URL
        return self._s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=self._expires_seconds,
        )

    def _s3(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            # 자격 증명은 넘기지 않는다 — boto3 기본 체인(환경변수·노드 IAM 역할 등)을 쓴다.
            # virtual: 버킷.s3.<리전>.amazonaws.com 주소로 만든다. 지정하지 않으면 boto3가
            # 리전 없는 전역 주소(버킷.s3.amazonaws.com)로 만드는데,
            # 리전 버킷은 그 주소에서 리다이렉트될 수 있다
            self._client = boto3.client(
                "s3",
                region_name=self._region or None,
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
            )
        return self._client

    def _object_key(self, value: str) -> str:
        """백엔드가 사진 경로를 어떤 형태로 저장했든 버킷 안 키로 바꾼다.
        키 그대로, gs://버킷/키, https://storage.googleapis.com/버킷/키, s3://버킷/키,
        https://버킷.s3.<리전>.amazonaws.com/키를 받는다.

        GCS 형태는 버킷 이름과 상관없이 키만 뗀다. s3일 때는 GCS 버킷 설정을 쓰지 않고,
        GCS에서 S3로 같은 키로 옮겼다고 보기 때문이다. S3 형태는 설정한 버킷일 때만 뗀다."""
        for prefix in ("gs://", "https://storage.googleapis.com/"):
            if value.startswith(prefix):
                bucket_and_key = value[len(prefix) :].split("/", 1)
                if len(bucket_and_key) == 2 and bucket_and_key[1]:
                    return bucket_and_key[1].split("?", 1)[0]
                return value
        s3_prefix = f"s3://{self._bucket}/"
        if value.startswith(s3_prefix):
            return value[len(s3_prefix) :].split("?", 1)[0]
        match = self._bucket_url.match(value)
        if match:
            return value[match.end() :].split("?", 1)[0]
        return value


class PassThroughPhotoSigner:
    """버킷 설정이 없을 때 — 키가 이미 https URL이면 그대로, 아니면 실패(재시도 대상)."""

    def __init__(self, bucket_setting: str = "GCS_PHOTO_BUCKET"):
        self._bucket_setting = bucket_setting

    def sign(self, key: str) -> str:
        if not key.startswith("https://"):
            raise ValueError(f"사진 키가 URL이 아닌데 {self._bucket_setting}이 설정되지 않았어요.")
        return key


def build_photo_signer(
    provider: Literal["gcs", "s3"],
    *,
    gcs_bucket: str,
    s3_bucket: str,
    region: str,
    minutes: int,
) -> PhotoSigner:
    """PHOTO_STORAGE_PROVIDER에 따라 고른다. gcs(기본)는 예전과 똑같이 동작한다."""
    if provider == "s3":
        if s3_bucket:
            return S3PhotoSigner(s3_bucket, region, minutes)
        return PassThroughPhotoSigner("S3_PHOTO_BUCKET")
    return GcsPhotoSigner(gcs_bucket, minutes) if gcs_bucket else PassThroughPhotoSigner()
