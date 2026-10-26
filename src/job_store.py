"""可重试的后台任务队列；同一主题只允许最新一代提交结果。"""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
from uuid import uuid4

from src.message_store import MessageStore


class JobStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        self._create_schema()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    user_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    generation INTEGER NOT NULL CHECK (generation >= 1),
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')
                    ),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
                    available_at TEXT NOT NULL,
                    lease_token TEXT,
                    lease_expires_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, job_id),
                    UNIQUE (user_id, kind, subject_key, generation)
                )
            """)

    @staticmethod
    def _time(value: str | None, name: str) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if not isinstance(value, str):
            raise ValueError(f"{name} 必须是带时区的时间字符串")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError(f"{name} 必须包含时区")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _json(value, name: str) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须能严格 JSON 编码") from error

    @staticmethod
    def _row(row) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["payload"] = json.loads(job.pop("payload_json"))
        result_json = job.pop("result_json")
        job["result"] = None if result_json is None else json.loads(result_json)
        return job

    def get_job(self, *, user_id: str, job_id: str) -> dict | None:
        self.message_store.validate_ids(user_id=user_id, job_id=job_id)
        row = self.conn.execute(
            "SELECT * FROM memory_jobs WHERE user_id = ? AND job_id = ?",
            (user_id, job_id),
        ).fetchone()
        return self._row(row)

    @contextmanager
    def _write_transaction(self):
        # 公开写入口各自提交；先结束旧读快照，再以写锁串行领取和分代检查。
        self.conn.commit()
        with self.conn:
            self.conn.execute("UPDATE memory_jobs SET attempts = attempts WHERE 0")
            yield

    def enqueue(
        self,
        *,
        user_id: str,
        job_id: str,
        kind: str,
        subject_key: str,
        payload,
        max_attempts: int = 3,
        available_at: str | None = None,
    ) -> dict:
        self.message_store.validate_ids(user_id=user_id, job_id=job_id)
        existing = self.get_job(user_id=user_id, job_id=job_id)
        if existing is not None:
            return existing
        self.message_store.assert_user_active(user_id=user_id)
        self.message_store.validate_ids(kind=kind, subject_key=subject_key)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts 必须是正整数")
        payload_json = self._json(payload, "payload")
        available = self._time(available_at, "available_at").isoformat()
        now = datetime.now(timezone.utc).isoformat()

        with self._write_transaction():
            self.message_store.assert_user_active(user_id=user_id)
            existing = self.get_job(user_id=user_id, job_id=job_id)
            if existing is not None:
                return existing
            row = self.conn.execute("""
                SELECT COALESCE(MAX(generation), 0) + 1 AS generation
                FROM memory_jobs
                WHERE user_id = ? AND kind = ? AND subject_key = ?
            """, (user_id, kind, subject_key)).fetchone()
            generation = row["generation"]
            self.conn.execute("""
                UPDATE memory_jobs
                SET status = 'cancelled', lease_token = NULL, lease_expires_at = NULL,
                    error = 'superseded by a newer generation', updated_at = ?
                WHERE user_id = ? AND kind = ? AND subject_key = ?
                  AND status IN ('pending', 'running')
            """, (now, user_id, kind, subject_key))
            if self.conn.execute("""
                SELECT 1 FROM memory_jobs
                WHERE user_id = ? AND kind = ? AND subject_key = ?
                  AND status IN ('pending', 'running') LIMIT 1
            """, (user_id, kind, subject_key)).fetchone() is not None:
                raise ValueError("旧一代任务未完整取消")
            cursor = self.conn.execute("""
                INSERT INTO memory_jobs (
                    user_id, job_id, kind, subject_key, generation, payload_json,
                    status, attempts, max_attempts, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
            """, (
                user_id, job_id, kind, subject_key, generation, payload_json,
                max_attempts, available, now, now,
            ))
            if cursor.rowcount != 1:
                raise ValueError("后台任务未保存")
        saved = self.get_job(user_id=user_id, job_id=job_id)
        assert saved is not None
        return saved

    def claim_next(
        self,
        *,
        user_id: str,
        lease_seconds: int = 60,
        now: str | None = None,
        lease_token: str | None = None,
        kind: str | None = None,
    ) -> dict | None:
        self.message_store.validate_ids(user_id=user_id)
        if kind is not None:
            self.message_store.validate_ids(kind=kind)
        if type(lease_seconds) is not int or lease_seconds < 1:
            raise ValueError("lease_seconds 必须是正整数")
        claimed_at = self._time(now, "now")
        token = str(uuid4()) if lease_token is None else lease_token
        self.message_store.validate_ids(lease_token=token)
        expires = (claimed_at + timedelta(seconds=lease_seconds)).isoformat()

        with self._write_transaction():
            conditions = ["user_id = ?", "status = 'pending'", "available_at <= ?"]
            parameters = [user_id, claimed_at.isoformat()]
            if kind is not None:
                conditions.append("kind = ?")
                parameters.append(kind)
            row = self.conn.execute(
                "SELECT job_id FROM memory_jobs WHERE " + " AND ".join(conditions)
                + " ORDER BY available_at, created_at, job_id LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            cursor = self.conn.execute("""
                UPDATE memory_jobs
                SET status = 'running', attempts = attempts + 1,
                    lease_token = ?, lease_expires_at = ?, updated_at = ?
                WHERE user_id = ? AND job_id = ? AND status = 'pending'
                  AND available_at <= ?
            """, (
                token, expires, claimed_at.isoformat(), user_id, row["job_id"],
                claimed_at.isoformat(),
            ))
            if cursor.rowcount != 1:
                raise ValueError("后台任务未成功领取")
        return self.get_job(user_id=user_id, job_id=row["job_id"])

    def complete(self, *, user_id: str, job_id: str, lease_token: str, result,
                 now: str | None = None) -> dict:
        self.message_store.validate_ids(
            user_id=user_id, job_id=job_id, lease_token=lease_token,
        )
        with self._write_transaction():
            current = self.get_job(user_id=user_id, job_id=job_id)
            if current is None:
                raise ValueError("后台任务不存在")
            if current["status"] == "succeeded" and current["lease_token"] == lease_token:
                return current
            instant = self._time(now, "now")
            if (current["lease_expires_at"] is None
                    or self._time(current["lease_expires_at"], "lease_expires_at") <= instant):
                raise ValueError("任务租约已过期，不能提交结果")
            result_json = self._json(result, "result")
            latest = self.conn.execute("""
                SELECT MAX(generation) AS generation FROM memory_jobs
                WHERE user_id = ? AND kind = ? AND subject_key = ?
            """, (user_id, current["kind"], current["subject_key"])).fetchone()
            if current["generation"] != latest["generation"]:
                raise ValueError("旧任务不能覆盖新一代结果")
            cursor = self.conn.execute("""
                UPDATE memory_jobs
                SET status = 'succeeded', result_json = ?, error = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE user_id = ? AND job_id = ? AND status = 'running'
                  AND lease_token = ?
            """, (result_json, instant.isoformat(), user_id, job_id, lease_token))
            if cursor.rowcount != 1:
                raise ValueError("任务租约已失效，不能提交结果")
        saved = self.get_job(user_id=user_id, job_id=job_id)
        assert saved is not None
        return saved

    def fail(
        self,
        *,
        user_id: str,
        job_id: str,
        lease_token: str,
        error: str,
        retry_delay_seconds: int = 0,
        now: str | None = None,
    ) -> dict:
        self.message_store.validate_ids(
            user_id=user_id, job_id=job_id, lease_token=lease_token,
        )
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error 必须是非空字符串")
        if type(retry_delay_seconds) is not int or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds 必须是非负整数")
        now = self._time(now, "now")
        with self._write_transaction():
            current = self.get_job(user_id=user_id, job_id=job_id)
            if (current is None or current["status"] != "running"
                    or current["lease_token"] != lease_token
                    or current["lease_expires_at"] is None
                    or self._time(current["lease_expires_at"], "lease_expires_at") <= now):
                raise ValueError("任务租约已失效，不能记录失败")
            status = "failed" if current["attempts"] >= current["max_attempts"] else "pending"
            available = (now + timedelta(seconds=retry_delay_seconds)).isoformat()
            cursor = self.conn.execute("""
                UPDATE memory_jobs
                SET status = ?, available_at = ?, lease_token = NULL,
                    lease_expires_at = NULL, error = ?, updated_at = ?
                WHERE user_id = ? AND job_id = ? AND status = 'running'
                  AND lease_token = ?
            """, (
                status, available, error, now.isoformat(), user_id, job_id, lease_token,
            ))
            if cursor.rowcount != 1:
                raise ValueError("任务租约已失效，不能记录失败")
        saved = self.get_job(user_id=user_id, job_id=job_id)
        assert saved is not None
        return saved

    def recover_expired(self, *, now: str | None = None) -> int:
        instant = self._time(now, "now").isoformat()
        with self.conn:
            cursor = self.conn.execute("""
                UPDATE memory_jobs
                SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                    lease_token = NULL, lease_expires_at = NULL,
                    error = 'lease expired', available_at = ?, updated_at = ?
                WHERE status = 'running' AND lease_expires_at <= ?
            """, (instant, instant, instant))
            return cursor.rowcount

    def list_jobs(self, *, user_id: str, status: str | None = None) -> list[dict]:
        self.message_store.validate_ids(user_id=user_id)
        if status is not None and (not isinstance(status, str) or status not in {
            "pending", "running", "succeeded", "failed", "cancelled",
        }):
            raise ValueError("status 不合法")
        if status is None:
            rows = self.conn.execute("""
                SELECT * FROM memory_jobs WHERE user_id = ?
                ORDER BY created_at, job_id
            """, (user_id,)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM memory_jobs WHERE user_id = ? AND status = ?
                ORDER BY created_at, job_id
            """, (user_id, status)).fetchall()
        return [self._row(row) for row in rows]


class BackgroundWorker:
    """一次领取并处理一项任务；业务处理器必须使用任务中的稳定操作身份。"""

    def __init__(self, job_store: JobStore, handlers: dict[str, object]):
        if not isinstance(handlers, dict) or any(
            not isinstance(kind, str) or not kind.strip() or not callable(handler)
            for kind, handler in handlers.items()
        ):
            raise ValueError("handlers 必须是 kind 到可调用对象的非空标识映射")
        self.job_store = job_store
        self.handlers = dict(handlers)

    def run_once(self, *, user_id: str, lease_seconds: int = 60) -> dict | None:
        job = self.job_store.claim_next(user_id=user_id, lease_seconds=lease_seconds)
        if job is None:
            return None
        handler = self.handlers.get(job["kind"])
        if handler is None:
            return self.job_store.fail(
                user_id=user_id, job_id=job["job_id"], lease_token=job["lease_token"],
                error=f"没有 {job['kind']} 处理器",
            )
        try:
            result = handler(job["payload"], job)
        except Exception as error:
            return self.job_store.fail(
                user_id=user_id, job_id=job["job_id"], lease_token=job["lease_token"],
                error=f"{type(error).__name__}: {error}",
            )
        return self.job_store.complete(
            user_id=user_id, job_id=job["job_id"], lease_token=job["lease_token"],
            result=result,
        )
