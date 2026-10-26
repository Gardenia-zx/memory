import unittest

from src.job_store import BackgroundWorker, JobStore
from src.message_store import MessageStore


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.messages = MessageStore(":memory:")
        self.addCleanup(self.messages.close)
        self.jobs = JobStore(self.messages)

    def test_enqueue_claim_complete_and_identity_retry(self):
        first = self.jobs.enqueue(
            user_id="u1", job_id="j1", kind="extract", subject_key="s1",
            payload={"seq": 2}, max_attempts=2, available_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(first["generation"], 1)
        self.assertEqual(self.jobs.enqueue(
            user_id="u1", job_id="j1", kind=None, subject_key=None, payload=float("nan"),
        ), first)
        claimed = self.jobs.claim_next(
            user_id="u1", lease_token="lease-1", now="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(claimed["attempts"], 1)
        saved = self.jobs.complete(
            user_id="u1", job_id="j1", lease_token="lease-1", result={"facts": 1},
            now="2026-01-01T00:00:01+00:00",
        )
        self.assertEqual(saved["status"], "succeeded")
        self.assertEqual(saved["result"], {"facts": 1})
        self.assertEqual(self.jobs.complete(
            user_id="u1", job_id="j1", lease_token="lease-1", result=None,
        ), saved)

    def test_new_generation_cancels_old_and_rejects_stale_completion(self):
        self.jobs.enqueue(user_id="u1", job_id="old", kind="extract", subject_key="s1", payload={})
        old = self.jobs.claim_next(user_id="u1", lease_token="old-lease")
        new = self.jobs.enqueue(
            user_id="u1", job_id="new", kind="extract", subject_key="s1", payload={"new": True},
        )
        self.assertEqual(new["generation"], 2)
        self.assertEqual(self.jobs.get_job(user_id="u1", job_id="old")["status"], "cancelled")
        with self.assertRaises(ValueError):
            self.jobs.complete(
                user_id="u1", job_id="old", lease_token=old["lease_token"], result={"stale": True},
            )
        claimed = self.jobs.claim_next(user_id="u1", lease_token="new-lease")
        self.assertEqual(claimed["job_id"], "new")

    def test_failure_retries_then_becomes_terminal_and_expired_leases_recover(self):
        self.jobs.enqueue(
            user_id="u1", job_id="retry", kind="extract", subject_key="a",
            payload={}, max_attempts=2, available_at="2026-01-01T00:00:00+00:00",
        )
        self.jobs.claim_next(user_id="u1", lease_token="l1", now="2026-01-01T00:00:00+00:00")
        pending = self.jobs.fail(
            user_id="u1", job_id="retry", lease_token="l1", error="temporary",
            now="2026-01-01T00:00:01+00:00",
        )
        self.assertEqual(pending["status"], "pending")
        self.jobs.claim_next(user_id="u1", lease_token="l2")
        failed = self.jobs.fail(
            user_id="u1", job_id="retry", lease_token="l2", error="again",
        )
        self.assertEqual(failed["status"], "failed")

        self.jobs.enqueue(
            user_id="u1", job_id="expired", kind="summary", subject_key="b", payload={},
            available_at="2026-01-01T00:00:00+00:00",
        )
        self.jobs.claim_next(
            user_id="u1", lease_token="l3", now="2026-01-01T00:00:00+00:00",
            lease_seconds=10,
        )
        self.assertEqual(self.jobs.recover_expired(now="2026-01-01T00:00:11+00:00"), 1)
        self.assertEqual(self.jobs.get_job(user_id="u1", job_id="expired")["status"], "pending")

    def test_users_and_kinds_are_isolated_and_invalid_leases_do_not_write(self):
        for user_id in ("u1", "u2"):
            self.jobs.enqueue(
                user_id=user_id, job_id="same", kind="extract", subject_key="same", payload={"u": user_id},
            )
        self.jobs.enqueue(user_id="u1", job_id="summary", kind="summary", subject_key="s", payload={})
        claimed = self.jobs.claim_next(user_id="u1", kind="summary", lease_token="summary-lease")
        self.assertEqual(claimed["job_id"], "summary")
        before = self.jobs.get_job(user_id="u1", job_id="summary")
        with self.assertRaises(ValueError):
            self.jobs.complete(user_id="u1", job_id="summary", lease_token="wrong", result={})
        self.assertEqual(self.jobs.get_job(user_id="u1", job_id="summary"), before)
        self.assertEqual(len(self.jobs.list_jobs(user_id="u1")), 2)
        self.assertEqual(len(self.jobs.list_jobs(user_id="u2")), 1)

    def test_worker_dispatches_handlers_and_persists_failures_for_retry(self):
        self.jobs.enqueue(
            user_id="u1", job_id="ok", kind="extract", subject_key="a", payload={"value": 2},
        )
        worker = BackgroundWorker(
            self.jobs, {"extract": lambda payload, job: {"value": payload["value"] * 2}},
        )
        saved = worker.run_once(user_id="u1")
        self.assertEqual(saved["status"], "succeeded")
        self.assertEqual(saved["result"], {"value": 4})

        self.jobs.enqueue(
            user_id="u1", job_id="bad", kind="unknown", subject_key="b", payload={},
            max_attempts=1,
        )
        failed = worker.run_once(user_id="u1")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("没有 unknown 处理器", failed["error"])

    def test_expired_lease_cannot_complete_or_fail_before_recovery_scans(self):
        self.jobs.enqueue(user_id="u1", job_id="j", kind="extract", subject_key="s", payload={},
                          available_at="2026-01-01T00:00:00Z")
        claimed = self.jobs.claim_next(user_id="u1", lease_token="expired", lease_seconds=10,
                                       now="2026-01-01T00:00:00Z")
        with self.assertRaises(ValueError):
            self.jobs.complete(user_id="u1", job_id="j", lease_token="expired", result={},
                               now="2026-01-01T00:00:10Z")
        with self.assertRaises(ValueError):
            self.jobs.fail(user_id="u1", job_id="j", lease_token="expired", error="late",
                           now="2026-01-01T00:00:11Z")
        self.assertEqual(self.jobs.get_job(user_id="u1", job_id="j"), claimed)
        self.assertEqual(self.jobs.recover_expired(now="2026-01-01T00:00:11Z"), 1)
        self.jobs.claim_next(user_id="u1", lease_token="new", now="2026-01-01T00:00:11Z")
        saved = self.jobs.complete(user_id="u1", job_id="j", lease_token="new", result={"ok": True},
                                   now="2026-01-01T00:00:12Z")
        self.assertEqual(self.jobs.complete(user_id="u1", job_id="j", lease_token="new", result=float("nan")), saved)

    def test_ignored_generation_cancellation_or_claim_rolls_back_trigger_side_effects(self):
        first = self.jobs.enqueue(user_id="u1", job_id="old", kind="extract", subject_key="s", payload={})
        for operation in ("cancelled", "running"):
            with self.messages.conn:
                self.messages.conn.execute(f"""
                    CREATE TRIGGER ignore_job BEFORE UPDATE ON memory_jobs
                    WHEN NEW.status = '{operation}' BEGIN
                        UPDATE memory_jobs SET error = 'side-effect' WHERE job_id = 'old';
                        SELECT RAISE(IGNORE); END
                """)
            try:
                with self.assertRaises(ValueError):
                    if operation == "cancelled":
                        self.jobs.enqueue(user_id="u1", job_id="new", kind="extract", subject_key="s", payload={})
                    else:
                        self.jobs.claim_next(user_id="u1")
                self.assertEqual(self.jobs.get_job(user_id="u1", job_id="old"), first)
                self.assertIsNone(self.jobs.get_job(user_id="u1", job_id="new"))
            finally:
                with self.messages.conn:
                    self.messages.conn.execute("DROP TRIGGER ignore_job")


if __name__ == "__main__":
    unittest.main()
