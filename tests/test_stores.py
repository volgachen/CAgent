import tempfile
import unittest
from unittest.mock import patch

from cagent.stores import MemoryStore, SQLiteStore


class StoreContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")

    def test_sessions_and_missing_ids(self):
        self.assertEqual(self.store.get_status("s"), "IDLE")
        self.store.set_status("s", "INTERRUPTED")
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")
        with self.assertRaises(ValueError):
            self.store.create_session("s", "u")
        with self.assertRaises(KeyError):
            self.store.list_messages("missing")

    def test_reads_and_writes_are_copies(self):
        original = {"role": "user", "content": "hello", "metadata": {"tags": ["a"]}}
        saved = self.store.save_messages("s", [original])
        original["metadata"]["tags"].append("changed")
        saved[0]["content"] = "changed"
        read = self.store.list_messages("s")
        self.assertEqual(read[0]["content"], "hello")
        self.assertEqual(read[0]["metadata"], {"tags": ["a"]})
        read[0]["metadata"].clear()
        self.assertEqual(self.store.list_messages("s")[0]["metadata"], {"tags": ["a"]})

    def test_atomic_batch_and_global_message_ids(self):
        self.store.create_session("other", "u")
        self.store.save_messages("other", [{"id": "occupied", "role": "user"}])
        with self.assertRaises(ValueError):
            self.store.save_messages("s", [{"id": "new", "role": "user"},
                                           {"id": "occupied", "role": "user"}])
        self.assertEqual(self.store.list_messages("s"), [])
        self.store.save_messages("s", [{"id": "new", "role": "user"}])
        with self.assertRaises(ValueError):
            self.store.save_messages("s", [{"id": "duplicate", "role": "user"}] * 2)

    def test_anchors_soft_deletes_and_stable_ties(self):
        stamp = "2026-09-14T00:00:00.000000+00:00"
        self.store.save_messages("s", [
            {"id": "b", "role": "user", "created_at": stamp},
            {"id": "a", "role": "user", "created_at": stamp},
        ])
        for name in ("m2", "m1"):
            self.store.save_messages("s", [{"id": name, "role": "mark", "name": "start",
                "created_at": stamp, "metadata": {"before_message_id": "b"}}])
        self.assertEqual([m["id"] for m in self.store.list_messages("s")], ["a", "m1", "m2", "b"])
        self.store.delete_message("s", "b")
        self.assertEqual([m["id"] for m in self.store.list_messages("s")], ["a", "m1", "m2"])
        self.assertEqual(len(self.store.list_messages("s", include_deleted=True)), 4)

    def test_invalid_anchors_do_not_write(self):
        self.store.create_session("other", "u")
        self.store.save_messages("other", [{"id": "foreign", "role": "user"}])
        self.store.save_messages("s", [{"id": "mark", "role": "mark", "name": "end"}])
        for target in ("missing", "foreign", "mark"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.store.save_messages("s", [{"role": "mark", "name": "start",
                    "metadata": {"before_message_id": target}}])
        self.assertEqual(len(self.store.list_messages("s")), 1)

    def test_tool_columns_and_json_are_preserved(self):
        call = {"id": "call", "name": "menu", "arguments": {"category": "咖啡"}}
        self.store.save_messages("s", [
            {"role": "assistant", "content": "我查一下", "tool_calls": [call]},
            {"role": "tool", "content": '{"items":["拿铁"]}', "tool_call_id": "call"},
        ])
        rows = self.store.list_messages("s")
        self.assertEqual(rows[0]["tool_calls"], [call])
        self.assertEqual(rows[0]["content"], "我查一下")
        self.assertEqual(rows[1]["tool_call_id"], "call")
        self.assertEqual(rows[1]["content"], '{"items":["拿铁"]}')

    def test_message_types_round_trip_without_inferring_from_role(self):
        self.store.save_messages("s", [
            {"role": "user", "type": "signal", "content": "用户输入"},
            {"role": "user", "type": "extra", "content": "额外指令"},
            {"role": "tool", "type": "memory", "content": "记忆", "tool_call_id": "m"},
            {"role": "user", "content": "未指定类型"},
        ])
        self.assertEqual([row["type"] for row in self.store.list_messages("s")],
                         ["signal", "extra", "memory", None])

    def test_invalid_messages_do_not_write(self):
        for bad in (
            {"role": "mark"}, {"role": "tool"}, {"role": "unknown"},
            {"role": "user", "session_id": "other"},
            {"role": "assistant", "tool_calls": [{"id": "c", "name": "x", "arguments": "{}"}]},
            {"role": "user", "metadata": {"bad": float("nan")}},
            {"role": "user", "type": "unknown"}, {"role": "user", "type": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.save_messages("s", [{"role": "user", "id": "valid"}, bad])
        self.assertEqual(self.store.list_messages("s"), [])

    def test_generated_timestamps_keep_append_order_when_clock_stalls(self):
        with patch("cagent.stores.now", return_value="2026-09-14T00:00:00.000000+00:00"):
            self.store.save_messages("s", [{"id": "z", "role": "user"}])
            self.store.save_messages("s", [{"id": "b", "role": "assistant"}, {"id": "a", "role": "user"}])
        self.assertEqual([m["id"] for m in self.store.list_messages("s")], ["z", "b", "a"])


class MemoryStoreTests(StoreContract, unittest.TestCase):
    def make_store(self):
        return MemoryStore()


class SQLiteStoreTests(StoreContract, unittest.TestCase):
    def make_store(self):
        store = SQLiteStore()
        self.addCleanup(store.close)
        store.initialize()
        return store

    def test_file_database_survives_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/agent.db"
            store = SQLiteStore(path)
            try:
                store.initialize()
                store.create_session("s", "u")
                store.save_messages("s", [{"role": "user", "type": "signal", "content": "保存"}])
            finally:
                store.close()
            reopened = SQLiteStore(path)
            try:
                self.assertEqual(reopened.list_messages("s")[0]["content"], "保存")
                self.assertEqual(reopened.list_messages("s")[0]["type"], "signal")
            finally:
                reopened.close()
