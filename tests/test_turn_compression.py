import tempfile
import unittest
from pathlib import Path

from cagent import BaseAgent, Tool
from cagent.stores import MemoryStore, SQLiteStore


BIG = "Shelf 3 holds 40 bags of Ethiopian beans and 12 bags of Colombian beans. " * 20


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools):
        return next(iterator)

    return model


def inventory(*, store, ctx, agent_cls):
    return BIG


def call(call_id):
    return {"id": call_id, "name": "inventory", "arguments": {}}


class InventoryAgent(BaseAgent):
    TURN_COMPRESSION_TOOL_CALL_THRESHOLD = 1
    COMPRESSION_TOKEN_THRESHOLD = None
    COMPRESSION_TURN_THRESHOLD = None

    @classmethod
    def get_tools(cls, ctx):
        return {"inventory": Tool(inventory, "Read inventory")}


class TurnCompressionContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")

    async def turn(self, model, agent=InventoryAgent, text="How many beans?", **settings):
        configured = type("Configured", (agent,), settings) if settings else agent
        return await configured.run_turn(text, session_id="s", store=self.store,
                                         model=model, pipeline=[])

    async def test_turn_with_tool_call_is_replaced_by_summary(self):
        requests = []

        async def summarizer(*, messages, tools):
            requests.append(messages)
            self.assertEqual(tools, [])
            self.assertEqual(self.store.get_status("s"), "RUNNING")
            return {"content": "User asked about beans; inventory shows 40 Ethiopian, 12 Colombian."}

        await self.turn(scripted({"content": "hi"}), text="hello")
        ctx = await self.turn(
            scripted({"tool_calls": [call("a")]}, {"content": "40 and 12 bags."}),
            COMPRESSION_MODEL=summarizer)
        self.assertTrue(ctx.turn_compression_applied)
        self.assertIsNone(ctx.turn_compression_error)
        self.assertEqual(ctx.reply, "40 and 12 bags.")

        # The summarizer saw earlier context, then the marker, then this turn.
        sent = requests[0]
        self.assertEqual(sent[0]["content"], BaseAgent.TURN_COMPRESSION_INSTRUCTION)
        contents = [m["content"] for m in sent]
        marker = contents.index(BaseAgent.TURN_COMPRESSION_MARKER)
        self.assertEqual(contents[marker - 2:marker], ["hello", "hi"])
        self.assertEqual(contents[marker + 1], "How many beans?")
        self.assertTrue(any(m["role"] == "tool" for m in sent[marker:]))

        # Everything after the marker, user input included, is now one summary.
        projected = ctx.model_messages()
        self.assertEqual([m["content"] for m in projected[1:3]], ["hello", "hi"])
        self.assertEqual(len(projected), 4)
        self.assertEqual(projected[3]["role"], "user")
        self.assertTrue(projected[3]["content"].startswith(BaseAgent.TURN_SUMMARY_PREFIX))

        # The temporary marker is gone and was never persisted.
        self.assertIsNone(ctx.turn_start_mark)
        self.assertFalse(any(m.get("name") == "tmp_turn_start_mark" for m in ctx.messages))
        rows = self.store.list_messages("s", include_deleted=True)
        self.assertFalse(any(m.get("name") == "tmp_turn_start_mark" for m in rows))
        live = self.store.list_messages("s")
        self.assertFalse(any(m["role"] == "tool" for m in live))
        self.assertEqual(live[-1]["name"], "turn_end")
        self.assertTrue(live[-1]["metadata"]["turn_compression_applied"])
        self.assertLess(live[-1]["metadata"]["num_tokens"],
                        live[-2]["metadata"]["num_tokens_before"])
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_below_threshold_or_disabled_keeps_turn(self):
        ctx = await self.turn(scripted({"content": "no tools"}), COMPRESSION_MODEL=scripted())
        self.assertFalse(ctx.turn_compression_applied)
        ctx = await self.turn(
            scripted({"tool_calls": [call("a")]}, {"content": "done"}),
            TURN_COMPRESSION_TOOL_CALL_THRESHOLD=2, COMPRESSION_MODEL=scripted())
        self.assertFalse(ctx.turn_compression_applied)
        ctx = await self.turn(
            scripted({"tool_calls": [call("b")]}, {"content": "done"}),
            TURN_COMPRESSION_TOOL_CALL_THRESHOLD=None, COMPRESSION_MODEL=scripted())
        self.assertFalse(ctx.turn_compression_applied)
        self.assertEqual(sum(m["role"] == "tool" for m in ctx.model_messages()), 2)
        self.assertIsNone(ctx.turn_start_mark)

    async def test_threshold_counts_all_calls_in_the_turn(self):
        ctx = await self.turn(
            scripted({"tool_calls": [call("a")]}, {"tool_calls": [call("b")]}, {"content": "done"}),
            TURN_COMPRESSION_TOOL_CALL_THRESHOLD=2,
            COMPRESSION_MODEL=scripted({"content": "Two inventory reads."}))
        self.assertTrue(ctx.turn_compression_applied)

    async def test_step_limit_turn_is_compressed(self):
        ctx = await self.turn(
            scripted({"tool_calls": [call("a")]}), MAX_STEPS=1,
            COMPRESSION_MODEL=scripted({"content": "Read inventory; no answer yet."}))
        self.assertEqual(ctx.stop_reason, "step_limit")
        self.assertTrue(ctx.turn_compression_applied)
        self.assertFalse(any(m["role"] == "tool" for m in ctx.model_messages()))

    async def test_interrupted_or_failed_turn_is_not_compressed(self):
        def interrupting(*, store, ctx, agent_cls):
            store.set_status(ctx.session_id, "INTERRUPTED")
            return BIG

        class Interrupting(InventoryAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"inventory": Tool(interrupting)}

        ctx = await self.turn(scripted({"tool_calls": [call("a")]}), agent=Interrupting,
                              COMPRESSION_MODEL=scripted())
        self.assertEqual(ctx.stop_reason, "interrupted")
        self.assertFalse(ctx.turn_compression_applied)
        self.assertIsNone(ctx.turn_start_mark)

        async def broken(*, messages, tools):
            raise RuntimeError("model down")

        self.store.set_status("s", "IDLE")
        with self.assertRaisesRegex(RuntimeError, "model down"):
            await self.turn(broken, COMPRESSION_MODEL=scripted())
        self.assertFalse(any(m.get("name") == "turn_summary"
                             for m in self.store.list_messages("s", include_deleted=True)))

    async def test_invalid_summary_keeps_original_turn(self):
        for output in ({"content": ""}, {"content": BIG * 5},
                       {"content": "s", "tool_calls": [call("x")]}):
            with self.subTest(output=str(output)[:30]):
                before = len(self.store.list_messages("s"))
                ctx = await self.turn(
                    scripted({"tool_calls": [call(f"c{before}")]}, {"content": "done"}),
                    COMPRESSION_MODEL=scripted(output))
                self.assertFalse(ctx.turn_compression_applied)
                self.assertIsNotNone(ctx.turn_compression_error)
                self.assertEqual(ctx.reply, "done")
                self.assertTrue(any(m["role"] == "tool" for m in ctx.model_messages()))
                self.assertIsNone(ctx.turn_start_mark)

    async def test_global_compression_runs_after_turn_compression(self):
        order = []

        async def summarizer(*, messages, tools):
            order.append(messages[0]["content"])
            return {"content": "Short summary."}

        await self.turn(scripted({"content": "noted"}), text=BIG)
        ctx = await self.turn(
            scripted({"tool_calls": [call("a")]}, {"content": "done"}),
            FORCE_COMPRESSION=True, COMPRESSION_MODEL=summarizer)
        self.assertEqual(order, [BaseAgent.TURN_COMPRESSION_INSTRUCTION,
                                 BaseAgent.COMPRESSION_INSTRUCTION])
        self.assertTrue(ctx.turn_compression_applied)
        self.assertTrue(ctx.compression_applied)

    async def test_next_turn_sees_summary_instead_of_tool_history(self):
        await self.turn(scripted({"tool_calls": [call("a")]}, {"content": "done"}),
                        COMPRESSION_MODEL=scripted({"content": "Beans: 40 and 12."}))
        seen = []

        async def model(*, messages, tools):
            seen.append(messages)
            return {"content": "ok"}

        await self.turn(model, text="Thanks")
        contents = [m["content"] for m in seen[0]]
        self.assertEqual(contents[1], BaseAgent.TURN_SUMMARY_PREFIX + "Beans: 40 and 12.")
        self.assertEqual(contents[2], "Thanks")

    async def test_threshold_is_validated(self):
        for bad in (0, -1, True, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await self.turn(scripted(), TURN_COMPRESSION_TOOL_CALL_THRESHOLD=bad)
        self.assertEqual(self.store.list_messages("s"), [])


class MemoryTurnCompressionTests(TurnCompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return MemoryStore()


class SQLiteTurnCompressionTests(TurnCompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        store = SQLiteStore()
        store.initialize()
        self.addCleanup(store.close)
        return store

    async def test_reopened_database_keeps_turn_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            store = SQLiteStore(path)
            store.initialize()
            store.create_session("p", "u")
            configured = type("C", (InventoryAgent,), {
                "COMPRESSION_MODEL": scripted({"content": "Beans: 40 and 12."})})
            ctx = await configured.run_turn(
                "Beans?", session_id="p", store=store, pipeline=[],
                model=scripted({"tool_calls": [call("a")]}, {"content": "done"}))
            self.assertTrue(ctx.turn_compression_applied)
            store.close()
            reopened = SQLiteStore(path)
            seen = []

            async def model(*, messages, tools):
                seen.append(messages)
                return {"content": "ok"}

            await BaseAgent.run_turn("Next", session_id="p", store=reopened, model=model,
                                     pipeline=[])
            reopened.close()
            self.assertEqual([m["content"] for m in seen[0][1:]],
                             [BaseAgent.TURN_SUMMARY_PREFIX + "Beans: 40 and 12.", "Next"])


if __name__ == "__main__":
    unittest.main()
