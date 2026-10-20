"""자물쇠 사진 서명 URL 생성 — 기능4 동기화가 백엔드 DB의 GCS 키로 사진 URL을 만든다.

V1에서는 백엔드가 서명 URL을 만들어 보내 줬지만, pull로 바꾸면서 AI가 직접 만든다.
GCP VM의 서비스 계정으로 서명하므로 키 파일이 필요 없고, 대신 그 서비스 계정에
'서비스 계정 토큰 생성자' 역할이 필요하다 (클라우드 팀 설정).
"""

from datetime import timedelta
from typing import Protocol


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


class PassThroughPhotoSigner:
    """버킷 설정이 없을 때 — 키가 이미 https URL이면 그대로, 아니면 실패(재시도 대상)."""

    def sign(self, key: str) -> str:
        if not key.startswith("https://"):
            raise ValueError("사진 키가 URL이 아닌데 GCS_PHOTO_BUCKET이 설정되지 않았어요.")
        return key


def build_photo_signer(bucket: str, minutes: int) -> PhotoSigner:
    return GcsPhotoSigner(bucket, minutes) if bucket else PassThroughPhotoSigner()
