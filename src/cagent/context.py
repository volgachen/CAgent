"""Mutable data for ONE turn, passed explicitly between hooks."""

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from .storage import Store, maybe_await


@dataclass
class Context:
    session_id: str
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[dict] = field(default_factory=list)
    num_tokens: int | None = 0
    token_count_is_estimate: bool = True
    token_count_error: str | None = None
    usage: list[dict] = field(default_factory=list)
    compression_token_threshold: int | None = 1000000
    compression_turn_threshold: int | None = 100
    force_compression: bool = False
    compression_applied: bool = False
    compression_error: str | None = None
    step_index: int = 0
    max_steps: int = 8
    stop_flag: bool = False
    stop_reason: str | None = None
    agent_name: str = "Agent"
    tools: dict = field(default_factory=dict)
    pending_messages: list[str] = field(default_factory=list)
    reply_content: str | None = None
    system_prompt: str = ""

    def add_messages(self, messages):
        self.messages.extend(deepcopy(messages))

    def stop(self, reason):
        self.stop_flag = True
        self.stop_reason = reason

    def active_history(self):
        """Return the committed summary and the records after its frozen boundary."""
        for message in reversed(self.messages):
            if (message["role"] == "mark" and message.get("name") == "compression_completed"
                    and message.get("deleted_at") is None):
                through_id = message["metadata"]["through_message_id"]
                for index, source in enumerate(self.messages):
                    if source["id"] == through_id:
                        return message["content"], self.messages[index + 1:]
                raise ValueError("Compression boundary is missing; history needs recovery")
        return None, self.messages

    @property
    def turns_since_compression(self):
        _, records = self.active_history()
        return sum(message["role"] == "mark" and message.get("name") == "turn_end"
                   and message.get("deleted_at") is None
                   and message["metadata"]["stop_reason"] == "completed"
                   for message in records)

    @staticmethod
    def summary_message(content):
        # Summary is conversation data, never promoted to a system instruction.
        return {"role": "user", "content": "Summary of earlier conversation:\n" + content}

    def model_messages(self):
        """Keep marks local and reject incomplete tool history before calling a model."""
        summary, records = self.active_history()
        result = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        if summary is not None:
            result.append(self.summary_message(summary))
        pending, seen = set(), set()
        for message in records:
            if message.get("deleted_at") is not None or message["role"] == "mark":
                continue
            role = message["role"]
            item = {"role": role, "content": message["content"]}
            if role == "tool":
                call_id = message["tool_call_id"]
                if call_id not in pending:
                    raise ValueError("Tool result has no pending call; history needs recovery")
                pending.remove(call_id)
                item["tool_call_id"] = call_id
            else:
                if pending:
                    raise ValueError("Unresolved tool calls; history needs recovery")
                calls = message.get("tool_calls", [])
                if calls:
                    item["tool_calls"] = calls
                    for call in calls:
                        if call["id"] in seen:
                            raise ValueError("Duplicate tool call id in history")
                        seen.add(call["id"])
                        pending.add(call["id"])
            result.append(item)
        if pending:
            raise ValueError("Unresolved tool calls; history needs recovery")
        return deepcopy(result)

    @property
    def reply(self):
        """The turn's final reply: a completion tool's text, else the last assistant text."""
        if self.stop_reason != "completed":
            return None
        if self.reply_content is not None:
            return self.reply_content or None
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                return message["content"] or None
        return None


async def append_messages(store: Store, ctx, messages):
    """Persist first; only then expose saved messages to the following step."""
    saved = await maybe_await(store.save_messages(ctx.session_id, messages))
    ctx.add_messages(saved)
    return saved
