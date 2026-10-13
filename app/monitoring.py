"""Prometheus 지표 — GET /metrics.

요청 수·응답 시간·상태 코드별 개수를 모아 /metrics로 내보낸다. 인프라(Prometheus)가 주기적으로
이 주소를 읽어 간다. /metrics는 /health처럼 내부 토큰 없이 열리므로 외부에 노출하지 않는
내부망에서만 접근해야 한다.

- 지표의 경로 이름은 `/api/embeddings/{recordId}`처럼 라우트 틀로 기록된다. 실제 ID가 지표에
  들어가지 않아서 개인정보도 없고 지표 종류가 끝없이 늘지도 않는다. 없는 주소로 오는 요청은
  한 묶음("none")으로 센다.
- 앱마다 별도 저장소(CollectorRegistry)를 쓴다. 기본 저장소를 공유하면 앱을 여러 번 만드는
  테스트에서 같은 지표 이름이 중복 등록되어 오류가 난다.
- 이 앱(gateway)의 지표만 낸다. moderation 서버와 worker는 따로 프로세스라 여기서 보이지 않는다.
- 프로세스(CPU·메모리) 지표는 리눅스에서만 나온다 (운영 컨테이너는 리눅스).
"""

from fastapi import FastAPI
from prometheus_client import CollectorRegistry, GCCollector, PlatformCollector, ProcessCollector
from prometheus_fastapi_instrumentator import Instrumentator

METRICS_PATH = "/metrics"
# 상태 확인과 지표 읽기 자체는 지표에서 뺀다 (자주 불려서 요청 수·응답 시간 그래프를 오염시킨다)
EXCLUDED_HANDLERS = ["/health", "/api/health", METRICS_PATH]


def setup_metrics(app: FastAPI) -> None:
    registry = CollectorRegistry()
    # 기본 저장소에 자동으로 붙는 프로세스(CPU·메모리)·파이썬·GC 지표를 이 저장소에도 붙인다
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)

    Instrumentator(excluded_handlers=EXCLUDED_HANDLERS, registry=registry).instrument(app).expose(
        app, endpoint=METRICS_PATH, include_in_schema=False
    )
