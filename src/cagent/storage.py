"""Storage contract implemented and supplied by the host application.

The framework does not create stores, open connections or manage transactions.
Implementations need not inherit Store. Methods may be synchronous or
asynchronous: the framework awaits awaitable results transparently, so a host
can expose an async database driver directly.
"""

import inspect
from typing import Protocol


async def maybe_await(value):
    """Await a store result when the implementation is asynchronous."""
    if inspect.isawaitable(value):
        return await value
    return value


class Store(Protocol):
    """Host-owned session and message access.

    Records are independent dictionaries, not live ORM entities. Missing session
    or message IDs raise KeyError; invalid input or duplicate IDs raise ValueError.
    The caller serializes turns for each session. Implementations may mix
    synchronous and asynchronous methods freely.

    Framework bookkeeping uses ordinary marks named turn_end, compression_started
    and compression_completed. Preserve their content/metadata unchanged; these
    names are reserved. Compression boundaries may refer to a turn_end mark,
    including a soft-deleted one. They use metadata.through_message_id (not the
    ordering anchor before_message_id). Saving a compression_completed mark
    atomically publishes its summary. No additional database-specific methods
    are required.
    """

    def create_session(self, session_id: str, user_id: str) -> dict:
        """Create a session with status IDLE; return its record."""
        ...

    def get_session(self, session_id: str) -> dict:
        """Return id, user_id, status, created_at, updated_at and deleted_at."""
        ...

    def get_status(self, session_id: str) -> str: ...

    def set_status(self, session_id: str, status: str) -> None: ...

    def list_messages(self, session_id: str, *, include_deleted: bool = False) -> list[dict]:
        """Return normalized records in model-history order.

        Sort by (created_at, id), placing marks with metadata.before_message_id
        before the referenced ordinary message in this session. Resolve anchors
        before filtering soft-deleted rows, so deleting a target keeps placement.
        """
        ...

    def save_messages(self, session_id: str, messages: list[dict]) -> list[dict]:
        """Atomically validate/save a batch; return saved copies in input order.

        Fill omitted fields: id, session_id, type=None, name=None, content="", metadata={},
        tool_calls=[], tool_call_id=None, from_session_id=None, created_at,
        deleted_at=None. A message's role is required. Roles are user, assistant,
        tool and mark; marks require a name, tools require a tool_call_id.

        type is a separate business classification: signal, memory, extra or None.
        Preserve the explicit value; do not infer it from role or tool names.
        Hosts typically label meaningful user input and final replies as signal,
        retrieved knowledge as memory, and supplemental instructions or tool
        errors as extra. Unclassified internal messages and legacy records may
        have type=None.

        tool_calls belongs to assistant messages and contains {id, name,
        arguments} dictionaries, with arguments an object and unique call IDs
        within the message. metadata and tool_calls must be JSON-compatible.

        IDs are globally unique. Anchors reference existing ordinary messages
        in the same session (including other rows in this batch). Reject invalid
        batches without partial writes. Generated timestamps must preserve append
        order across writes/restarts; normalize explicit timestamps to UTC.
        """
        ...

    def delete_message(self, session_id: str, message_id: str) -> None:
        """Soft-delete a message in this session, retaining its anchor position."""
        ...
