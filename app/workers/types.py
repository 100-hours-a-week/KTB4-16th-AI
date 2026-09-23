from collections.abc import Awaitable, Callable
from typing import Any

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]
