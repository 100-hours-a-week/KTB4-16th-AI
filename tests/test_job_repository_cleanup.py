from types import SimpleNamespace

from app.db.repositories.job_repository import JobRepository


class CapturingSession:
    def __init__(self, rowcount=3):
        self.rowcount = rowcount
        self.executed: list[tuple[str, dict]] = []
        self.commits = 0

    async def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return SimpleNamespace(rowcount=self.rowcount)

    async def commit(self):
        self.commits += 1


async def test_only_finished_jobs_of_the_given_type_older_than_the_limit_are_deleted():
    session = CapturingSession(rowcount=3)

    deleted = await JobRepository(session).delete_finished(
        job_type="record_facts_sync", older_than_hours=48
    )

    sql, params = session.executed[0]
    assert "DELETE FROM ai_jobs" in sql
    assert "job_type = :job_type" in sql  # 다른 종류의 작업은 건드리지 않는다
    assert "status IN ('done', 'failed', 'cancelled')" in sql  # 대기·실행 중인 작업은 남긴다
    assert "updated_at < now() - make_interval(hours => :hours)" in sql
    assert params == {"job_type": "record_facts_sync", "hours": 48}
    assert deleted == 3 and session.commits == 1
