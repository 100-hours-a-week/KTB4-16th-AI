from collections.abc import Awaitable, Callable
from typing import Any

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]

# job이 최종적으로 끝난 뒤(성공, 또는 재시도를 다 써서 포기) 한 번 부르는 후처리
JobFinishedHook = Callable[[dict[str, Any]], Awaitable[None]]
