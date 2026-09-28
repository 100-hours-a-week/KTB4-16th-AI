"""백엔드 MySQL 커넥션 (읽기 전용) — 기능2 RECAP이 records를 조회하는 데 사용.

MySQL 8 기본 인증(caching_sha2_password)은 암호화 안 된 채널에서는 매번
RSA 공개키 교환이 필요한데, aiomysql이 non-SSL 연결에서 이걸 못 해 자격증명이
맞아도 Access denied가 난다(실측). 그래서 SSL을 강제로 켠다 — 원격 IP로 붙는
내부 서버라 CA 인증서 체인은 없어서 인증서 검증 자체는 끈다(전송 구간
암호화만 필요, 서버 신원 검증까지는 불필요).
"""

import ssl
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(
        get_settings().backend_mysql_url,
        pool_pre_ping=True,
        connect_args={"ssl": _ssl_context()},
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session
