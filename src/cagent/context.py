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
    # Per-turn cache of resolved image references (ref -> URL), so a turn
    # resolves each stored image at most once across steps.
    resolved_images: dict = field(default_factory=dict)
    # Keep only the most recent N tool-result images in model input (None:
    # keep all). Older ones are replaced by a text note; user images are
    # never trimmed.
    tool_image_limit: int | None = None
    # Turn-scoped compression: the threshold on tool calls made within one
    # turn (None disables it) and the in-memory marker of where the turn
    # started. The marker is never persisted and is removed when the turn ends.
    turn_compression_tool_call_threshold: int | None = None
    turn_start_mark: dict | None = None
    turn_compression_applied: bool = False
    turn_compression_error: str | None = None

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

    def turn_messages(self):
        """Records after this turn's start marker (empty without a marker)."""
        for index, message in enumerate(self.messages):
            if message is self.turn_start_mark:
                return self.messages[index + 1:]
        return []

    def turn_tool_call_count(self):
        return sum(len(message.get("tool_calls") or []) for message in self.turn_messages()
                   if message["role"] == "assistant" and message.get("deleted_at") is None)

    @staticmethod
    def replaced_message_ids(records):
        """IDs hidden by a saved turn summary; its single write publishes the replacement."""
        hidden = set()
        for message in records:
            if (message["role"] == "user" and message.get("name") == "turn_summary"
                    and message.get("deleted_at") is None):
                hidden.update(message.get("metadata", {}).get("replaced_message_ids") or [])
        return hidden

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

    @staticmethod
    def interleave(text, images):
        """Rebuild ordered parts: an image's "at" offset places it inside the text;
        images without "at" (plain images= input) follow all text."""
        parts, cursor = [], 0
        for image in images:
            at = image.get("at", len(text))
            if isinstance(at, bool) or not isinstance(at, int) or not cursor <= at <= len(text):
                raise ValueError("Image offsets must be ordered positions within the text")
            if text[cursor:at]:
                parts.append({"type": "text", "text": text[cursor:at]})
            parts.append({"type": "image_ref", **{k: v for k, v in image.items() if k != "at"}})
            cursor = at
        if text[cursor:]:
            parts.append({"type": "text", "text": text[cursor:]})
        return parts

    @staticmethod
    def omitted_tool_images_note(count):
        return f"\n[{count} earlier tool image(s) omitted]"

    def _kept_tool_images(self, records):
        """(message id, image index) pairs of the newest tool images within the limit."""
        if self.tool_image_limit is None:
            return None
        kept, budget = set(), self.tool_image_limit
        for message in reversed(records):
            if budget <= 0:
                break
            if (message["role"] != "tool" or message.get("deleted_at") is not None):
                continue
            images = message.get("metadata", {}).get("images") or []
            for index in range(len(images) - 1, -1, -1):
                if budget <= 0:
                    break
                kept.add((message["id"], index))
                budget -= 1
        return kept

    def model_messages(self):
        """Keep marks local and reject incomplete tool history before calling a model."""
        summary, records = self.active_history()
        hidden = self.replaced_message_ids(records)
        if hidden:
            records = [message for message in records if message["id"] not in hidden]
        result = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        if summary is not None:
            result.append(self.summary_message(summary))
        kept_tool_images = self._kept_tool_images(records)
        pending, seen = set(), set()
        for message in records:
            if message.get("deleted_at") is not None or message["role"] == "mark":
                continue
            role = message["role"]
            item = {"role": role, "content": message["content"]}
            images = message.get("metadata", {}).get("images") if role in ("user", "tool") else None
            if images and role == "tool" and kept_tool_images is not None:
                kept = [image for index, image in enumerate(images)
                        if (message["id"], index) in kept_tool_images]
                if len(kept) < len(images):
                    item["content"] += self.omitted_tool_images_note(len(images) - len(kept))
                images = kept
            if images:
                # Stored references only; BaseAgent.prepare_model_messages resolves
                # them into image_url parts right before a model call.
                item["content"] = self.interleave(item["content"], images)
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
