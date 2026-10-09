"""Reference Store implementations: in-memory for tests, SQLite for demos.

Production services should implement the Store protocol against their own
database; nothing here is required by the framework core.
"""

import json
import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


SCHEMA = """
CREATE TABLE sessions (
    id TEXT NOT NULL PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'IDLE',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT DEFAULT NULL
);

CREATE TABLE messages (
    id TEXT NOT NULL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    type TEXT DEFAULT NULL CHECK (type IS NULL OR type IN ('signal', 'memory', 'extra')),
    name TEXT DEFAULT NULL,
    content TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_call_id TEXT DEFAULT NULL,
    from_session_id TEXT DEFAULT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT DEFAULT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    CHECK (role IN ('user', 'assistant', 'tool', 'mark')),
    CHECK (role <> 'mark' OR (name IS NOT NULL AND LENGTH(TRIM(name)) > 0)),
    CHECK (
        (role = 'tool' AND tool_call_id IS NOT NULL AND LENGTH(TRIM(tool_call_id)) > 0)
        OR (role <> 'tool' AND tool_call_id IS NULL)
    )
);

CREATE INDEX messages_session_created_id ON messages(session_id, created_at, id);
CREATE INDEX messages_session_tool_call_id ON messages(session_id, tool_call_id);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


def _prepare(session_id, message):
    row = dict(id=uuid4().hex, session_id=session_id, role=None, type=None, name=None,
               content="", metadata={}, tool_calls=[], tool_call_id=None,
               from_session_id=None, created_at=now(), deleted_at=None)
    if set(message) - set(row):
        raise ValueError("Unknown message fields")
    row.update(deepcopy(message))
    if row["session_id"] != session_id:
        raise ValueError("Message belongs to another session")
    for field in ("id", "created_at"):
        _text(row[field], field)
    stamp = datetime.fromisoformat(row["created_at"])
    if stamp.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    row["created_at"] = stamp.astimezone(timezone.utc).isoformat(timespec="microseconds")
    for field in ("name", "from_session_id", "deleted_at"):
        if row[field] is not None:
            _text(row[field], field)
    role = row["role"]
    if role not in ("user", "assistant", "tool", "mark"):
        raise ValueError("Invalid message role")
    if row["type"] not in (None, "signal", "memory", "extra"):
        raise ValueError("Invalid message type")
    if not isinstance(row["content"], str) or not isinstance(row["metadata"], dict):
        raise ValueError("content must be text and metadata must be an object")
    if role == "mark":
        _text(row["name"], "mark name")
    if "before_message_id" in row["metadata"]:
        if role != "mark":
            raise ValueError("Only marks may specify before_message_id")
        _text(row["metadata"]["before_message_id"], "before_message_id")
    if role == "tool":
        _text(row["tool_call_id"], "tool_call_id")
    elif row["tool_call_id"] is not None:
        raise ValueError("Only tool messages have tool_call_id")
    calls = row["tool_calls"]
    if not isinstance(calls, list) or (calls and role != "assistant"):
        raise ValueError("tool_calls must be an assistant's list")
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
            raise ValueError("A tool call requires id, name and arguments")
        _text(call["id"], "call id")
        _text(call["name"], "tool name")
        if call["id"] in ids or not isinstance(call["arguments"], dict):
            raise ValueError("Tool call IDs must be unique and arguments must be an object")
        ids.add(call["id"])
    for field in ("metadata", "tool_calls"):
        encoded = json.dumps(row[field], ensure_ascii=False, allow_nan=False)
        if json.loads(encoded) != row[field]:
            raise ValueError(f"{field} must be JSON data with string object keys")
    return row


def _ordered(rows, include_deleted=False):
    """Resolve marks against all rows, including soft-deleted anchor targets."""
    rows = sorted(rows, key=lambda row: (row["created_at"], row["id"]))
    by_id = {row["id"]: row for row in rows}
    before = {}
    for row in rows:
        target = row["metadata"].get("before_message_id")
        if target is not None:
            if target not in by_id or by_id[target]["role"] == "mark":
                raise ValueError("Mark anchor must reference an ordinary message in the same session")
            before.setdefault(target, []).append(row)
    ordered = []
    for row in rows:
        if "before_message_id" not in row["metadata"]:
            ordered.extend(before.get(row["id"], []))
            ordered.append(row)
    return deepcopy([row for row in ordered if include_deleted or row["deleted_at"] is None])


def _batch(session_id, messages, existing):
    rows = [_prepare(session_id, message) for message in messages]
    session_rows = [row for row in existing if row["session_id"] == session_id]
    latest = max((row["created_at"] for row in session_rows), default="")
    for message, row in zip(messages, rows):
        # Persisted timestamps preserve append order even on coarse system clocks.
        if "created_at" not in message and row["created_at"] <= latest:
            row["created_at"] = (datetime.fromisoformat(latest) + timedelta(microseconds=1)).isoformat(timespec="microseconds")
        latest = max(latest, row["created_at"])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) & {row["id"] for row in existing}:
        raise ValueError("Duplicate message id")
    _ordered(session_rows + rows, True)
    return rows


class MemoryStore:
    """In-memory store for tests and demos; hold one instance across turns."""

    def __init__(self):
        self._sessions = {}
        self._messages = {}

    def create_session(self, session_id, user_id):
        _text(session_id, "session_id")
        _text(user_id, "user_id")
        if session_id in self._sessions:
            raise ValueError("Duplicate session id")
        stamp = now()
        self._sessions[session_id] = dict(id=session_id, user_id=user_id, status="IDLE",
                                          created_at=stamp, updated_at=stamp, deleted_at=None)
        return self.get_session(session_id)

    def get_session(self, session_id):
        return deepcopy(self._sessions[session_id])

    def get_status(self, session_id):
        return self.get_session(session_id)["status"]

    def set_status(self, session_id, status):
        _text(status, "status")
        self._sessions[session_id].update(status=status, updated_at=now())

    def list_messages(self, session_id, *, include_deleted=False):
        self.get_session(session_id)
        return _ordered([row for row in self._messages.values() if row["session_id"] == session_id], include_deleted)

    def save_messages(self, session_id, messages):
        self.get_session(session_id)
        rows = _batch(session_id, messages, list(self._messages.values()))
        self._messages.update({row["id"]: row for row in rows})
        return deepcopy(rows)

    def delete_message(self, session_id, message_id):
        self.get_session(session_id)
        row = self._messages[message_id]
        if row["session_id"] != session_id:
            raise KeyError(message_id)
        row["deleted_at"] = row["deleted_at"] or now()


class SQLiteStore:
    """SQLite reference store: one connection, used on its creating thread, short writes."""

    def __init__(self, path=":memory:"):
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def initialize(self):
        """Create empty tables explicitly; never migrate existing tables implicitly."""
        self._connection.executescript(SCHEMA)

    def close(self):
        self._connection.close()

    def create_session(self, session_id, user_id):
        _text(session_id, "session_id")
        _text(user_id, "user_id")
        stamp = now()
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO sessions (id,user_id,created_at,updated_at) VALUES (?,?,?,?)",
                    (session_id, user_id, stamp, stamp))
        except sqlite3.IntegrityError as exc:
            raise ValueError("Duplicate session id") from exc
        return self.get_session(session_id)

    def get_session(self, session_id):
        row = self._connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return dict(row)

    def get_status(self, session_id):
        return self.get_session(session_id)["status"]

    def set_status(self, session_id, status):
        _text(status, "status")
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE sessions SET status=?,updated_at=? WHERE id=?", (status, now(), session_id))
            if not cursor.rowcount:
                raise KeyError(session_id)

    def _read_messages(self, session_id):
        rows = self._connection.execute(
            "SELECT * FROM messages WHERE session_id=?", (session_id,)).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            for field in ("metadata", "tool_calls"):
                row[field] = json.loads(row[field])
            result.append(row)
        return result

    def list_messages(self, session_id, *, include_deleted=False):
        self.get_session(session_id)
        return _ordered(self._read_messages(session_id), include_deleted)

    def save_messages(self, session_id, messages):
        with self._connection:
            # Acquire the write transaction before validating anchor and ID constraints.
            self._connection.execute("BEGIN IMMEDIATE")
            self.get_session(session_id)
            rows = _batch(session_id, messages, self._read_messages(session_id))
            for row in rows:
                if self._connection.execute("SELECT 1 FROM messages WHERE id=?", (row["id"],)).fetchone():
                    raise ValueError("Duplicate message id")
                values = dict(row)
                for field in ("metadata", "tool_calls"):
                    values[field] = json.dumps(values[field], ensure_ascii=False, allow_nan=False)
                self._connection.execute(
                    "INSERT INTO messages (id,session_id,role,type,name,content,metadata,tool_calls,tool_call_id,"
                    "from_session_id,created_at,deleted_at) VALUES (:id,:session_id,:role,:type,:name,:content,"
                    ":metadata,:tool_calls,:tool_call_id,:from_session_id,:created_at,:deleted_at)", values)
        return rows

    def delete_message(self, session_id, message_id):
        self.get_session(session_id)
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE messages SET deleted_at=COALESCE(deleted_at,?) WHERE session_id=? AND id=?",
                (now(), session_id, message_id))
            if not cursor.rowcount:
                raise KeyError(message_id)
