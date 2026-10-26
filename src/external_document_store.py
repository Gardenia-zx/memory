"""Versioned external text and stable character chunks, separate from chat evidence."""

from datetime import datetime, timezone
import json

from src.message_store import MessageStore


def _time(value, name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


class ExternalDocumentStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.conn = message_store.conn
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS external_documents (
                user_id TEXT NOT NULL, document_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK(version >= 1),
                content TEXT NOT NULL, metadata_json TEXT NOT NULL,
                scope TEXT NOT NULL CHECK(scope IN ('user', 'session')),
                session_id TEXT, valid_from TEXT, valid_to TEXT,
                status TEXT NOT NULL CHECK(status IN ('active','superseded','retracted')),
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(user_id, document_id, version),
                CHECK((scope='user' AND session_id IS NULL) OR
                      (scope='session' AND session_id IS NOT NULL AND length(trim(session_id))>0))
            )""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS external_document_chunks (
                user_id TEXT NOT NULL, document_id TEXT NOT NULL, version INTEGER NOT NULL,
                chunk_id TEXT NOT NULL, start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL, text TEXT NOT NULL,
                PRIMARY KEY(user_id, document_id, version, chunk_id),
                FOREIGN KEY(user_id, document_id, version)
                    REFERENCES external_documents(user_id, document_id, version)
            )""")

    def get_document(self, *, user_id, document_id, version=1):
        MessageStore.validate_ids(user_id=user_id, document_id=document_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(version) is not int or version < 1:
            raise ValueError("version must be a positive integer")
        row = self.conn.execute("""SELECT * FROM external_documents
            WHERE user_id=? AND document_id=? AND version=?""",
            (user_id, document_id, version)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["chunks"] = [dict(chunk) for chunk in self.conn.execute("""
            SELECT * FROM external_document_chunks
            WHERE user_id=? AND document_id=? AND version=? ORDER BY start_offset""",
            (user_id, document_id, version)).fetchall()]
        return result

    def get_latest_document(self, *, user_id, document_id):
        MessageStore.validate_ids(user_id=user_id, document_id=document_id)
        self.message_store.assert_user_active(user_id=user_id)
        row = self.conn.execute("""SELECT MAX(version) AS version FROM external_documents
            WHERE user_id=? AND document_id=?""", (user_id, document_id)).fetchone()
        return None if row["version"] is None else self.get_document(
            user_id=user_id, document_id=document_id, version=row["version"])

    def get_chunk(self, *, user_id, document_id, version, chunk_id):
        MessageStore.validate_ids(chunk_id=chunk_id)
        document = self.get_document(user_id=user_id, document_id=document_id, version=version)
        if document is None:
            return None
        return next((chunk for chunk in document["chunks"] if chunk["chunk_id"] == chunk_id), None)

    def add_document(self, *, user_id, document_id, content=None, version=1,
                     metadata=None, scope="user", session_id=None,
                     valid_from=None, valid_to=None, chunk_size=800, overlap=100):
        MessageStore.validate_ids(user_id=user_id, document_id=document_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(version) is not int or version < 1:
            raise ValueError("version must be a positive integer")
        with self.conn:
            existing = self.get_document(user_id=user_id, document_id=document_id, version=version)
            if existing is not None:
                return existing
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be nonempty text")
            if type(chunk_size) is not int or chunk_size < 1:
                raise ValueError("chunk_size must be a positive integer")
            if type(overlap) is not int or not 0 <= overlap < chunk_size:
                raise ValueError("overlap must be an integer smaller than chunk_size")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            metadata_json = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
            if scope == "user":
                if session_id is not None:
                    raise ValueError("user documents cannot specify session_id")
            elif scope == "session":
                MessageStore.validate_ids(session_id=session_id)
            else:
                raise ValueError("scope must be user or session")
            start, end = _time(valid_from, "valid_from"), _time(valid_to, "valid_to")
            if start is not None and end is not None and datetime.fromisoformat(end) <= datetime.fromisoformat(start):
                raise ValueError("valid_to must follow valid_from")
            latest = self.get_latest_document(user_id=user_id, document_id=document_id)
            if latest is None:
                if version != 1:
                    raise ValueError("the first document version must be 1")
            elif latest["status"] != "active" or version != latest["version"] + 1:
                raise ValueError("new versions must follow the latest active version")
            if latest is not None:
                changed = self.conn.execute("""UPDATE external_documents SET status='superseded'
                    WHERE user_id=? AND document_id=? AND version=? AND status='active'""",
                    (user_id, document_id, latest["version"]))
                if changed.rowcount != 1:
                    raise ValueError("document status changed")
            cursor = self.conn.execute("""INSERT INTO external_documents
                (user_id,document_id,version,content,metadata_json,scope,session_id,
                 valid_from,valid_to,status,recorded_at) VALUES (?,?,?,?,?,?,?,?,?,'active',?)""",
                (user_id, document_id, version, content, metadata_json, scope, session_id,
                 start, end, datetime.now(timezone.utc).isoformat()))
            if cursor.rowcount != 1:
                raise ValueError("document was not saved")
            offset, index = 0, 0
            while offset < len(content):
                chunk_end = min(offset + chunk_size, len(content))
                cursor = self.conn.execute("""INSERT INTO external_document_chunks
                    (user_id,document_id,version,chunk_id,start_offset,end_offset,text)
                    VALUES (?,?,?,?,?,?,?)""",
                    (user_id, document_id, version, f"chunk:{index:06d}",
                     offset, chunk_end, content[offset:chunk_end]))
                if cursor.rowcount != 1:
                    raise ValueError("document chunk was not saved")
                if chunk_end == len(content):
                    break
                offset, index = chunk_end - overlap, index + 1
            return self.get_document(user_id=user_id, document_id=document_id, version=version)

    def retract_document(self, *, user_id, document_id, expected_version):
        MessageStore.validate_ids(user_id=user_id, document_id=document_id)
        self.message_store.assert_user_active(user_id=user_id)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive integer")
        with self.conn:
            latest = self.get_latest_document(user_id=user_id, document_id=document_id)
            if latest is None or latest["version"] != expected_version:
                raise ValueError("document version changed or does not exist")
            if latest["status"] == "retracted":
                return latest
            cursor = self.conn.execute("""UPDATE external_documents SET status='retracted'
                WHERE user_id=? AND document_id=? AND version=? AND status='active'""",
                (user_id, document_id, expected_version))
            if cursor.rowcount != 1:
                raise ValueError("document is not active")
            return self.get_document(user_id=user_id, document_id=document_id, version=expected_version)

    def list_chunks(self, *, user_id, session_id=None, as_of=None):
        MessageStore.validate_ids(user_id=user_id)
        self.message_store.assert_user_active(user_id=user_id)
        if session_id is not None:
            MessageStore.validate_ids(session_id=session_id)
        moment = _time(as_of, "as_of") or datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""SELECT c.*, d.metadata_json, d.scope, d.session_id,
            d.valid_from, d.valid_to FROM external_document_chunks c JOIN external_documents d
            USING(user_id,document_id,version) WHERE d.user_id=? AND d.status='active'
            AND (d.scope='user' OR (d.scope='session' AND d.session_id=?))
            ORDER BY c.document_id,c.version,c.start_offset""", (user_id, session_id)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if ((item["valid_from"] is not None and datetime.fromisoformat(moment) < datetime.fromisoformat(item["valid_from"]))
                    or (item["valid_to"] is not None and datetime.fromisoformat(moment) >= datetime.fromisoformat(item["valid_to"]))):
                continue
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result
