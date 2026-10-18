"""텍스트 임베딩 서비스(embedding_main) 내부 API. gateway·worker가 같은 컨테이너 안에서만 부른다."""

from pydantic import Field, field_validator

from app.schemas.common import CamelModel

# 한 번에 보낼 수 있는 글 개수. 클라이언트가 이 크기로 나눠서 보낸다
MAX_TEXTS_PER_REQUEST = 64


class EmbedRequest(CamelModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS_PER_REQUEST)

    @field_validator("texts")
    @classmethod
    def _no_blank(cls, texts: list[str]) -> list[str]:
        # 빈 글은 벡터가 의미 없다. 호출하는 쪽(Embedder)도 막지만 서비스에서도 한 번 더 막는다
        if any(not t.strip() for t in texts):
            raise ValueError("빈 텍스트는 임베딩할 수 없어요.")
        return texts


class EmbedResponse(CamelModel):
    model: str
    dim: int
    vectors: list[list[float]]
