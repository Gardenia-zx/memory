from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.forgetter import MemoryForgetter
from src.message_store import MessageStore


class MessageStoreConcurrencyTests(unittest.TestCase):
    def disk_path(self):
        directory = Path(".test-tmp").resolve()
        directory.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=directory)

    @staticmethod
    def append_many(path, prefix):
        store = MessageStore(path)
        try:
            for index in range(20):
                store.append_message(
                    user_id="u1", session_id="shared", message_id=f"{prefix}-{index}",
                    occurred_at="2026-01-01T00:00:00+00:00",
                    payload={"role": "user", "content": f"{prefix}-{index}"},
                )
        finally:
            store.close()

    def test_two_connections_allocate_unique_contiguous_session_sequences(self):
        with self.disk_path() as directory:
            path = Path(directory) / "memory.sqlite"
            MessageStore(path).close()
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.append_many, path, prefix) for prefix in ("a", "b")]
                for future in futures:
                    future.result()
            store = MessageStore(path)
            try:
                records = store.list_messages(user_id="u1", session_id="shared")
                self.assertEqual(len(records), 40)
                self.assertEqual([record["seq"] for record in records], list(range(1, 41)))
            finally:
                store.close()

    def test_concurrent_same_identity_returns_one_first_record_without_sequence_gap(self):
        with self.disk_path() as directory:
            path = Path(directory) / "memory.sqlite"
            MessageStore(path).close()

            def append(content):
                store = MessageStore(path)
                try:
                    return store.append_message(
                        user_id="u1", session_id="shared", message_id="same",
                        occurred_at="2026-01-01T00:00:00+00:00",
                        payload={"role": "user", "content": content},
                    )
                finally:
                    store.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                records = list(executor.map(append, ("first", "second")))
            self.assertEqual(records[0], records[1])
            store = MessageStore(path)
            try:
                saved = store.list_messages(user_id="u1", session_id="shared")
                self.assertEqual(len(saved), 1)
                next_record = store.append_message(
                    user_id="u1", session_id="shared", message_id="next",
                    occurred_at="2026-01-01T00:00:01+00:00",
                    payload={"role": "user", "content": "next"},
                )
                self.assertEqual(next_record["seq"], 2)
            finally:
                store.close()

    def test_forget_between_first_read_and_write_cannot_resurrect_message(self):
        with self.disk_path() as directory:
            path = Path(directory) / "memory.sqlite"
            store = MessageStore(path)
            other = MessageStore(path)
            try:
                get_message = store.get_message
                forgotten = False

                def interleave(**fields):
                    nonlocal forgotten
                    record = get_message(**fields)
                    if not forgotten:
                        store.conn.commit()
                        MemoryForgetter(other).forget_user(user_id="u1", reason="interleaved delete")
                        forgotten = True
                    return record

                with patch.object(store, "get_message", side_effect=interleave):
                    with self.assertRaises(ValueError):
                        store.append_message(user_id="u1", session_id="s", message_id="late",
                                             occurred_at="2026-01-01T00:00:00Z",
                                             payload={"role": "user", "content": "must not return"})
                self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = 'u1'").fetchone()[0], 0)
                self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM message_sequences WHERE user_id = 'u1'").fetchone()[0], 0)
            finally:
                store.close()
                other.close()


if __name__ == "__main__":
    unittest.main()
