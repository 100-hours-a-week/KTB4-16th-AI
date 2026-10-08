"""기능6 모더레이션 — 채팅 한 줄의 유해 여부 판정 (동기 호출).

백엔드 흐름: 단어 필터(욕설 단어 ** 가리기, 스팸 차단) → 이 API → 결과에 따라 전송.
장애(503)·타임아웃이면 백엔드는 막는 쪽(fail-safe)으로 처리한다 (위키 모델 API 설계 6-2).
모델은 욕 없이 기분 나쁜 말(모욕·혐오·성희롱·협박)을 잡는 역할이고, 유해/정상 이진 판정만 한다.
"""

from starlette.concurrency import run_in_threadpool

from app.components.moderation_model import ModerationModel
from app.exceptions import ModelUnavailableError
from app.schemas.moderation import ModerationCheckRequest, ModerationCheckResponse


class ModerationService:
    def __init__(self, model: ModerationModel | None, threshold: float):
        self._model = model
        self._threshold = threshold

    async def check(self, req: ModerationCheckRequest) -> ModerationCheckResponse:
        if self._model is None:
            raise ModelUnavailableError("모더레이션 모델 파일이 없어요.")
        # 추론은 CPU 작업(수 ms)이라 이벤트 루프를 막지 않게 스레드에서 돌린다
        prob = await run_in_threadpool(self._model.toxic_probability, req.text)
        is_toxic = prob >= self._threshold
        return ModerationCheckResponse(
            request_id=req.request_id,
            is_toxic=is_toxic,
            # 유해 확률 (명세 6-2: 통과 예시 0.03, 차단 예시 0.94)
            confidence=round(prob, 4),
            # 분류별 판정은 아직 학습하지 않아 비워 둔다 (V3 다중 라벨). 분기는 isToxic으로
            category=None,
            model_version=self._model.version,
        )
