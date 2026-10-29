import unittest
from datetime import datetime, timedelta, timezone

from cagent import BaseAgent, Context, Tool
from cagent.stores import MemoryStore, SQLiteStore


LONG = "Guest A ordered a large latte. Guest B asked for an Americano. " * 30
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def counter(*, messages, tools, tool_choice=None):
    return sum(len(str(m["content"])) + 5 for m in messages) + len(tools) * 10


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools, tool_choice=None):
        return next(iterator)

    return model


def lookup(*, store, ctx, agent_cls):
    return LONG


class Quiet(BaseAgent):
    COMPRESSION_TOKEN_THRESHOLD = None
    COMPRESSION_TURN_THRESHOLD = None


class ToolAgent(Quiet):
    @classmethod
    def get_tools(cls, ctx):
        return {"lookup": Tool(lookup, "Look something up")}


class RangeCompressionContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")

    async def at(self, hours, awaitable):
        """Await, then move the rows it saved to T0 + hours (keeping their order)."""
        before = {m["id"] for m in self.store.list_messages("s", include_deleted=True)}
        result = await awaitable
        new = [m for m in self.store.list_messages("s", include_deleted=True)
               if m["id"] not in before]
        for offset, message in enumerate(new):
            self.set_time(message["id"], T0 + timedelta(hours=hours, microseconds=offset))
        return result

    async def turn(self, text, hours, reply="ok", agent=Quiet, model=None):
        await self.at(hours, agent.run_turn(text, session_id="s", store=self.store, pipeline=[],
                                            model=model or scripted({"content": reply})))

    async def three_turns(self):
        await self.turn("old " + LONG, 0, "old reply")
        await self.turn("mid " + LONG, 10, "mid reply")
        await self.turn("new " + LONG, 20, "new reply")

    async def compress(self, since, model=None, hours=25, **kwargs):
        return await self.at(hours, BaseAgent.compress_session(
            session_id="s", store=self.store, count_tokens=counter,
            model=model or scripted({"content": "Range summary"}), since=since, **kwargs))

    def contents(self, ctx):
        return [m["content"] for m in ctx.model_messages() if m["role"] != "system"]

    async def test_range_replaces_turns_after_since_and_keeps_older_history(self):
        await self.three_turns()
        requests = []

        async def summarizer(*, messages, tools, tool_choice=None):
            requests.append(messages)
            return {"content": "Range summary"}

        ctx = await self.compress(T0 + timedelta(hours=5), model=summarizer)
        self.assertTrue(ctx.compression_applied)
        self.assertEqual(ctx.compression_kind, "range")
        self.assertIsNone(ctx.compression_error)
        self.assertEqual(self.store.get_status("s"), "IDLE")

        # Summarizer: earlier history, marker, the range, then the instruction.
        sent = [m["content"] for m in requests[0]]
        marker = sent.index(BaseAgent.RANGE_COMPRESSION_MARKER)
        self.assertEqual(sent[marker - 2:marker], ["old " + LONG, "old reply"])
        self.assertEqual(sent[marker + 1:-1], ["mid " + LONG, "mid reply", "new " + LONG, "new reply"])
        self.assertEqual(sent[-1], BaseAgent.RANGE_COMPRESSION_INSTRUCTION)

        # The model now sees old history unchanged plus one summary.
        fresh = await Quiet.run_turn("next", session_id="s", store=self.store, pipeline=[],
                                     model=scripted({"content": "fine"}))
        self.assertEqual(self.contents(fresh)[:4], [
            "old " + LONG, "old reply",
            BaseAgent.RANGE_SUMMARY_PREFIX + "Range summary", "next"])
        summary = next(m for m in self.store.list_messages("s") if m.get("name") == "range_summary")
        self.assertEqual(summary["role"], "user")
        self.assertEqual(len(summary["metadata"]["replaced_message_ids"]), 4)
        self.assertLess(summary["metadata"]["num_tokens_after"], summary["metadata"]["num_tokens_before"])
        # Originals are soft-deleted; turn_end marks stay.
        visible = [m["content"] for m in self.store.list_messages("s")]
        self.assertNotIn("mid reply", visible)
        self.assertEqual(sum(m.get("name") == "turn_end" for m in self.store.list_messages("s")), 4)

    async def test_since_inside_a_turn_starts_at_the_next_turn(self):
        await self.three_turns()
        # The mid turn began at hour 10, before since, so it stays.
        ctx = await self.compress((T0 + timedelta(hours=10, seconds=1)).isoformat())
        self.assertEqual(ctx.compression_kind, "range")
        contents = self.contents(ctx)
        self.assertIn("mid reply", contents)
        self.assertNotIn("new reply", contents)

    async def test_custom_instruction_replaces_the_default(self):
        await self.three_turns()
        seen = []

        async def summarizer(*, messages, tools, tool_choice=None):
            seen.append(messages[-1])
            return {"content": "Range summary"}

        await self.compress(T0 + timedelta(hours=5), model=summarizer, instruction="Only list orders.")
        await BaseAgent.compress_session(session_id="s", store=self.store, model=summarizer,
                                         count_tokens=counter, instruction="Whole, briefly.")
        self.assertEqual(seen, [{"role": "user", "content": "Only list orders."},
                                {"role": "user", "content": "Whole, briefly."}])

    async def test_since_at_history_start_falls_back_to_whole_compression(self):
        await self.three_turns()
        ctx = await self.compress(T0 - timedelta(hours=1))
        self.assertEqual(ctx.compression_kind, "session")
        self.assertEqual(self.contents(ctx)[0],
                         "Summary of earlier conversation:\nRange summary")

    async def test_since_before_active_boundary_falls_back_to_whole_compression(self):
        await self.turn("old " + LONG, 0, "old reply")
        await self.at(1, BaseAgent.compress_session(
            session_id="s", store=self.store, count_tokens=counter,
            model=scripted({"content": "First summary"})))
        await self.turn("mid " + LONG, 10, "mid reply")
        await self.turn("new " + LONG, 20, "new reply")
        # Hour 5 lies inside the compressed part; nothing older remains to keep.
        ctx = await self.compress(T0 + timedelta(hours=5))
        self.assertEqual(ctx.compression_kind, "session")
        # A later range keeps the turns after that summary.
        await self.turn("between " + LONG, 27, "between reply")
        await self.turn("later " + LONG, 30, "later reply")
        ctx = await self.compress(T0 + timedelta(hours=28), hours=31)
        self.assertEqual(ctx.compression_kind, "range")

    async def test_nothing_after_since_is_skipped(self):
        await self.three_turns()
        ctx = await self.compress(T0 + timedelta(hours=30))
        self.assertFalse(ctx.compression_applied)
        self.assertEqual(ctx.compression_skip_reason, "no completed turns since the given time")
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_range_with_tool_calls_and_earlier_turn_summary(self):
        await self.turn("old " + LONG, 0, "old reply")

        class Squash(ToolAgent):
            TURN_COMPRESSION_TOOL_CALL_THRESHOLD = 1
            COMPRESSION_MODEL = scripted({"content": "Turn digest"})

        await self.turn("look " + LONG, 10, agent=Squash, model=scripted(
            {"tool_calls": [{"id": "a", "name": "lookup", "arguments": {}}]}, {"content": "found"}))
        await self.turn("new " + LONG, 20, "new reply", agent=ToolAgent, model=scripted(
            {"tool_calls": [{"id": "b", "name": "lookup", "arguments": {}}]}, {"content": "done"}))
        ctx = await self.compress(T0 + timedelta(hours=5))
        self.assertEqual(ctx.compression_kind, "range", ctx.compression_error)
        contents = self.contents(ctx)
        self.assertEqual(contents, ["old " + LONG, "old reply",
                                    BaseAgent.RANGE_SUMMARY_PREFIX + "Range summary"])

    async def test_later_range_after_a_range_summary(self):
        await self.three_turns()
        await self.compress(T0 + timedelta(hours=5), model=scripted({"content": "First range"}))
        await self.turn("next " + LONG, 30, "next reply")
        # since falls after the first range summary (hour 25): only the next turn.
        ctx = await self.compress(T0 + timedelta(hours=26), hours=31,
                                  model=scripted({"content": "Second range"}))
        self.assertEqual(ctx.compression_kind, "range", ctx.compression_error)
        self.assertEqual(self.contents(ctx), [
            "old " + LONG, "old reply", BaseAgent.RANGE_SUMMARY_PREFIX + "First range",
            BaseAgent.RANGE_SUMMARY_PREFIX + "Second range"])
        # since before it: the first summary is folded into a new one.
        await self.turn("last " + LONG, 40, "last reply")
        ctx = await self.compress(T0 + timedelta(hours=24), hours=41,
                                  model=scripted({"content": "Merged"}))
        self.assertEqual(ctx.compression_kind, "range", ctx.compression_error)
        self.assertEqual(self.contents(ctx), [
            "old " + LONG, "old reply", BaseAgent.RANGE_SUMMARY_PREFIX + "Merged"])

    async def test_invalid_arguments(self):
        await self.three_turns()
        for kwargs in ({"since": datetime(2026, 1, 1)}, {"since": "yesterday"},
                       {"since": T0, "instruction": " "}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await BaseAgent.compress_session(session_id="s", store=self.store,
                                                 model=scripted(), **kwargs)

    async def test_failed_summary_keeps_history(self):
        await self.three_turns()
        ctx = await self.compress(T0 + timedelta(hours=5), model=scripted({"content": LONG * 20}))
        self.assertFalse(ctx.compression_applied)
        self.assertIn("did not reduce", ctx.compression_error)
        self.assertIn("mid reply", self.contents(ctx))
        self.assertIn("mid reply", [m["content"] for m in self.store.list_messages("s")])


class MemoryRangeCompressionTests(RangeCompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return MemoryStore()

    def set_time(self, message_id, stamp):
        self.store._messages[message_id]["created_at"] = stamp.isoformat(timespec="microseconds")


class SQLiteRangeCompressionTests(RangeCompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        store = SQLiteStore()
        store.initialize()
        self.addCleanup(store.close)
        return store

    def set_time(self, message_id, stamp):
        with self.store._connection:
            self.store._connection.execute("UPDATE messages SET created_at=? WHERE id=?",
                                          (stamp.isoformat(timespec="microseconds"), message_id))


if __name__ == "__main__":
    unittest.main()
