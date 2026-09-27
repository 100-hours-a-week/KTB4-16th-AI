"""플레이리스트 저장용 뮤로 서비스 계정 OAuth — 딱 1번만 실행.

B안(서비스 계정 + 팔로우 링크): 사용자마다 OAuth를 받는 게 아니라, 뮤로 팀 소유
Spotify 계정 하나에만 1회 로그인해서 refresh_token을 받아둔다. 이후 앱은 이
refresh_token으로 계속 access_token을 갱신하며, 그 계정 안에 추천 플레이리스트를
만들고 사용자에게는 공유 링크만 준다(사용자는 "팔로우"로 자기 라이브러리에 추가).

사용법:
    1. .env에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET이 있어야 함
    2. python scripts/spotify_service_account_setup.py
    3. 출력된 URL을 "뮤로 공식 계정으로 쓸 Spotify 계정"으로 로그인해서 열기
       (개인 계정 말고, 팀이 공용으로 관리할 계정 — 새로 만들어도 됨)
    4. 로그인 후 리다이렉트된 주소창 URL 전체를 복사해서 여기에 붙여넣기
       (그 페이지 자체는 "연결할 수 없음" 에러가 떠도 정상 — 주소창의 code만 필요함)
    5. 출력된 SPOTIFY_SERVICE_REFRESH_TOKEN 값을 .env에 붙여넣기

주의: refresh_token은 그 계정의 플레이리스트를 계속 만들 수 있는 열쇠다.
      절대 커밋하거나 채팅에 붙여넣지 말고 .env에만 보관할 것.
"""

import base64
import sys
import urllib.parse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings  # noqa: E402

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = "playlist-modify-public playlist-modify-private"
REDIRECT_URI = "http://127.0.0.1:8888/callback"  # 대시보드에 이 값 등록 필요, 서버 안 띄워도 됨


def main() -> None:
    s = get_settings()
    if not s.spotify_client_id or not s.spotify_client_secret:
        sys.exit(".env에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET을 먼저 넣어 주세요.")

    params = {
        "client_id": s.spotify_client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "show_dialog": "true",
    }
    print("\n[1] 아래 URL을 뮤로 공식 계정으로 로그인해서 여세요 (개인 계정 아님):\n")
    print(f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}\n")
    print("※ developer.spotify.com 앱 설정의 Redirect URIs에 다음을 미리 등록해야 함:")
    print(f"  {REDIRECT_URI}\n")

    redirected = input("[2] 로그인 후 리다이렉트된 주소창 URL 전체를 붙여넣으세요: ").strip()
    query = urllib.parse.urlparse(redirected).query
    code = urllib.parse.parse_qs(query).get("code", [None])[0]
    if not code:
        sys.exit("URL에서 code를 못 찾았어요. 전체 주소를 다시 복사해 주세요.")

    basic = base64.b64encode(f"{s.spotify_client_id}:{s.spotify_client_secret}".encode()).decode()
    res = httpx.post(
        TOKEN_URL,
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
        headers={"Authorization": f"Basic {basic}"},
    )
    if res.status_code >= 400:
        sys.exit(f"토큰 발급 실패 ({res.status_code}): {res.text}")

    body = res.json()
    me = httpx.get(
        "https://api.spotify.com/v1/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    ).json()

    print(f"\n[완료] 서비스 계정: {me.get('display_name')} ({me.get('id')})")
    print("\n아래 줄을 .env에 붙여넣으세요 (채팅에는 붙여넣지 마세요):\n")
    print(f"SPOTIFY_SERVICE_REFRESH_TOKEN={body['refresh_token']}\n")


if __name__ == "__main__":
    main()
