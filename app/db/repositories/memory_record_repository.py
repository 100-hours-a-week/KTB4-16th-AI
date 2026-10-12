"""기능5 기억 검색 — 찾은 자물쇠의 상세를 백엔드 MySQL(읽기 전용)에서 읽는다.

AI DB에는 벡터와 곡 ID만 있고 날짜·장소·곡명은 백엔드 DB에 있다. 리포트(RECAP)가 이미 쓰는
읽기 전용 연결로 읽으므로 백엔드 코드는 안 바뀐다.

사용자 격리: SQL에 user_id 조건을 넣어서 다른 사람의 자물쇠는 id를 알아도 읽히지 않는다.
삭제된 자물쇠(deleted_at)는 뺀다 — AI DB 벡터는 백엔드가 삭제 호출을 보낼 때까지 남을 수 있다.

테이블·컬럼명은 백엔드 Flyway 마이그레이션 기준이다. records·places의 컬럼은 리포트 코드로 확인했고,
music_tracks의 title·album_image_url·external_url은 응답 스키마 주석(백엔드 컬럼과 같게 맞춤)을
근거로 한 것이라 실제 DB로 한 번 확인이 필요하다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class MemoryRecord:
    record_id: int
    created_at: datetime
    title: str
    artist_name: str
    external_track_id: str
    album_image_url: str
    external_url: str
    place_id: int | None
    place_name: str | None  # 백엔드 places.legal_dong_name — 비어 있을 수 있다


class MemoryRecordSource(Protocol):
    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]: ...


class MemoryRecordRepository:
    def __init__(self, mysql_session: AsyncSession):
        self._mysql = mysql_session

    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]:
        if not record_ids:
            return {}
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.record_id, r.created_at,
                       m.title, m.artist_name, m.external_track_id,
                       m.album_image_url, m.external_url,
                       p.place_id, p.legal_dong_name AS place_name
                FROM records r
                JOIN music_tracks m ON r.music_track_id = m.music_track_id
                JOIN places p ON r.place_id = p.place_id
                WHERE r.user_id = :user_id AND r.deleted_at IS NULL
                  AND r.record_id IN :record_ids
                """
            ).bindparams(bindparam("record_ids", expanding=True)),
            {"user_id": user_id, "record_ids": record_ids},
        )
        return {
            row.record_id: MemoryRecord(
                record_id=row.record_id,
                created_at=row.created_at,
                title=row.title,
                artist_name=row.artist_name,
                external_track_id=row.external_track_id,
                album_image_url=row.album_image_url or "",
                external_url=row.external_url or "",
                place_id=row.place_id,
                place_name=row.place_name,
            )
            for row in rows
        }
