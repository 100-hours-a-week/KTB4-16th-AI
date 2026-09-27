"""기능1 플레이리스트 저장 — B안: 뮤로 서비스 계정에 만들고 공유 링크 반환.

사용자별 OAuth 없이, 팀이 소유한 Spotify 계정 하나(SpotifyServiceAccountClient)에
플레이리스트를 만든다. 사용자는 반환된 링크에서 "팔로우"하면 자기 라이브러리에도
나타난다 (Development Mode 25명 제한과 무관, 확장성 이슈 없음).
"""

from app.clients.spotify_client import SpotifyPlaylistClient
from app.schemas.context_recommend import PlaylistSaveRequest, PlaylistSaveResponse
from app.services.not_implemented import not_implemented


class PlaylistService:
    def __init__(self, spotify_playlist: SpotifyPlaylistClient):
        self._spotify_playlist = spotify_playlist

    async def save(self, req: PlaylistSaveRequest) -> PlaylistSaveResponse:
        if req.platform != "spotify":
            # YouTube 대체 연동(FR-012)은 Google OAuth가 별도로 필요 — 미결정, 추후 구현
            raise not_implemented("기능1 YouTube 플레이리스트 저장")

        playlist_url = await self._spotify_playlist.create_playlist_with_tracks(
            title=req.title,
            track_uris=req.track_uris,
            description=f"뮤로가 추천한 플레이리스트 (userId: {req.user_id})",
        )
        return PlaylistSaveResponse(playlist_url=playlist_url, platform="spotify")
