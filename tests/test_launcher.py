import sys

from app.launcher import supervise


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_one_process_dies_stops_the_rest_and_returns_its_code():
    """하나라도 죽으면 반쯤 살아있는 컨테이너로 남지 않게 전체를 내린다."""
    code = supervise(
        {
            "long": _py("import time; time.sleep(60)"),
            "crash": _py("import sys, time; time.sleep(0.2); sys.exit(3)"),
        }
    )

    assert code == 3


def test_process_exiting_normally_is_still_treated_as_failure():
    """계속 떠 있어야 하는 프로세스가 코드 0으로 끝나도 비정상으로 본다."""
    code = supervise({"long": _py("import time; time.sleep(60)"), "quit": _py("pass")})

    assert code == 1
