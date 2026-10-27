import asyncio
import tempfile
import unittest
from pathlib import Path

from cagent import BaseAgent, Context
from cagent.stores import MemoryStore, SQLiteStore


LONG = "Guest A ordered a large latte. Guest B asked for an Americano. " * 30


def counter(*, messages, tools):
    # Deterministic stand-in for a model-specific tokenizer.
    return sum(len(m["content"]) + 5 for m in messages) + len(tools) * 10


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools):
        return next(iterator)

    return model


class CompressionContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")

    async def turn(self, **kwargs):
        settings = {key.upper(): kwargs.pop(key) for key in (
            "compression_token_threshold", "compression_turn_threshold", "compression_model",
            "force_compression", "count_tokens") if key in kwargs}
        configured = type("ConfiguredAgent", (BaseAgent,), settings)
        return await configured.run_turn(
            kwargs.pop("input_message", LONG), session_id="s", store=self.store, pipeline=[],
            model=kwargs.pop("model", scripted({"content": "ok"})), **kwargs)

    async def test_threshold_turns_rolling_summary_and_original_reply(self):
        requests = []

        async def summarizer(*, messages, tools):
            requests.append(messages)
            self.assertEqual(tools, [])
            # Same prefix as a normal model call, instruction last as a user turn.
            self.assertEqual(messages[0], {"role": "system", "content": BaseAgent.SYSTEM_PROMPT})
            self.assertEqual(messages[-1], {"role": "user", "content": BaseAgent.COMPRESSION_INSTRUCTION})
            self.assertEqual(sum(m["role"] == "system" for m in messages), 1)
            self.assertEqual(self.store.get_status("s"), "RUNNING")
            return {"content": f"Summary {len(requests)}"}

        options = dict(compression_token_threshold=None, compression_turn_threshold=2,
                       compression_model=summarizer, count_tokens=counter)
        first = await self.turn(**options)
        self.assertEqual(first.turns_since_compression, 1)
        self.assertFalse(first.compression_applied)
        second = await self.turn(**options)
        self.assertTrue(second.compression_applied)
        self.assertEqual(second.turns_since_compression, 0)
        self.assertEqual(second.reply, "ok")
        self.assertEqual(second.model_messages(), [
            {"role": "system", "content": BaseAgent.SYSTEM_PROMPT},
            Context.summary_message("Summary 1")])
        before = second.messages[-1]["metadata"]["num_tokens_before"]
        self.assertLess(second.num_tokens, before)
        await self.turn(**options)
        fourth = await self.turn(**options)
        self.assertTrue(fourth.compression_applied)
        self.assertEqual(requests[1][1], Context.summary_message("Summary 1"))
        self.assertEqual(sum(m["content"] == LONG for m in requests[1]), 2)
        self.assertEqual(sum(m["role"] == "user" for m in self.store.list_messages("s")), 4)
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_token_threshold_is_inclusive_and_usage_is_separate(self):
        final_tokens = counter(messages=[{"content": BaseAgent.SYSTEM_PROMPT},
                                         {"content": LONG}, {"content": "ok"}], tools=[])
        ctx = await self.turn(
            compression_token_threshold=final_tokens, compression_turn_threshold=None,
            model=scripted({"content": "ok", "usage": {"input_tokens": 9999999}}),
            compression_model=scripted({"content": "Summary"}), count_tokens=counter)
        self.assertTrue(ctx.compression_applied)
        mark = next(m for m in ctx.messages if m.get("name") == "turn_end")
        self.assertEqual(mark["metadata"]["num_tokens"], final_tokens)
        self.assertEqual(mark["metadata"]["usage"], [{"input_tokens": 9999999}])
        self.assertFalse(mark["metadata"]["token_count_is_estimate"])

    async def test_external_compression_does_not_create_a_turn_or_repeat(self):
        await self.turn(compression_token_threshold=None, compression_turn_threshold=None)
        original_turns = sum(m.get("name") == "turn_end" for m in self.store.list_messages("s"))
        result = await BaseAgent.compress_session(
            session_id="s", store=self.store, model=scripted({"content": "Summary"}))
        self.assertTrue(result.compression_applied)
        result = await BaseAgent.compress_session(
            session_id="s", store=self.store, model=scripted())
        self.assertFalse(result.compression_applied)
        self.assertEqual(sum(m.get("name") == "turn_end" for m in self.store.list_messages("s")),
                         original_turns)

    async def test_summary_request_reuses_prefix_and_ends_with_user(self):
        # Providers such as Bedrock Claude reject a trailing assistant message
        # (prefill); the instruction must arrive as the final user message,
        # after the same system prompt + history a normal call would send.
        await self.turn(compression_token_threshold=None, compression_turn_threshold=None)
        requests = []

        async def summarizer(*, messages, tools):
            requests.append((messages, tools))
            return {"content": "Summary"}

        result = await BaseAgent.compress_session(
            session_id="s", store=self.store, model=summarizer, system_prompt="Persona X")
        self.assertTrue(result.compression_applied)
        messages, tools = requests[0]
        self.assertEqual(tools, [])
        self.assertEqual(messages[0], {"role": "system", "content": "Persona X"})
        self.assertEqual([m["role"] for m in messages[1:]], ["user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], LONG)
        self.assertEqual(messages[-1], {"role": "user", "content": BaseAgent.COMPRESSION_INSTRUCTION})

    async def test_invalid_summary_keeps_original_history_and_allows_retry(self):
        for output in ({"content": ""}, {"content": LONG * 10},
                       {"content": "summary", "tool_calls": [{"id": "forbidden"}]}):
            with self.subTest(output=str(output)[:40]):
                ctx = await self.turn(force_compression=True, compression_model=scripted(output))
                self.assertFalse(ctx.compression_applied)
                self.assertIsNotNone(ctx.compression_error)
                self.assertEqual(ctx.reply, "ok")
                self.assertTrue(any(m["content"] == LONG for m in ctx.model_messages()))
        retried = await BaseAgent.compress_session(
            session_id="s", store=self.store, model=scripted({"content": "Recovered summary"}))
        self.assertTrue(retried.compression_applied)

    async def test_fixed_boundary_keeps_later_messages(self):
        await self.turn(compression_token_threshold=None, compression_turn_threshold=None)
        self.store.save_messages("s", [{"role": "user", "content": "Arrived after finished turn"}])

        async def summary(*, messages, tools):
            self.assertFalse(any(m["content"] == "Arrived after finished turn" for m in messages))
            return {"content": "Summary"}

        ctx = await BaseAgent.compress_session(session_id="s", store=self.store, model=summary)
        self.assertTrue(ctx.compression_applied)
        self.assertEqual(ctx.model_messages()[-1]["content"], "Arrived after finished turn")

    async def test_soft_deleted_boundary_still_loads_summary(self):
        first = await self.turn(force_compression=True,
                                compression_model=scripted({"content": "Summary"}))
        self.store.delete_message("s", first.turn_id)
        seen = []

        async def model(*, messages, tools):
            seen.extend(messages)
            return {"content": ""}

        second = await self.turn(model=model, compression_token_threshold=None,
                                 compression_turn_threshold=None)
        self.assertEqual(seen[1], Context.summary_message("Summary"))
        self.assertIsNone(second.reply)

    async def test_summary_provider_failure_does_not_fail_successful_turn(self):
        async def broken(*, messages, tools):
            raise RuntimeError("summary provider unavailable")

        ctx = await self.turn(force_compression=True, compression_model=broken)
        self.assertEqual(ctx.stop_reason, "completed")
        self.assertEqual(ctx.reply, "ok")
        self.assertIn("summary provider unavailable", ctx.compression_error)
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_interruption_during_summary_keeps_original(self):
        async def summary(*, messages, tools):
            self.store.set_status("s", "INTERRUPTED")
            return {"content": "Summary"}

        ctx = await self.turn(force_compression=True, compression_model=summary)
        self.assertEqual(ctx.stop_reason, "interrupted")
        self.assertIsNone(ctx.reply)
        self.assertFalse(ctx.compression_applied)
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")
        self.assertFalse(any(m.get("name") == "compression_completed" for m in ctx.messages))

    async def test_cancellation_during_summary_propagates_without_commit(self):
        started = asyncio.Event()

        async def waiting(*, messages, tools):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(self.turn(force_compression=True, compression_model=waiting))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")
        self.assertFalse(any(m.get("name") == "compression_completed"
                             for m in self.store.list_messages("s")))

    async def test_failed_summary_write_does_not_activate_in_memory(self):
        real_save = self.store.save_messages

        def save(session_id, messages):
            if any(m.get("name") == "compression_completed" for m in messages):
                raise RuntimeError("write failed")
            return real_save(session_id, messages)

        self.store.save_messages = save
        ctx = await self.turn(force_compression=True,
                              compression_model=scripted({"content": "Summary"}))
        self.assertFalse(ctx.compression_applied)
        self.assertEqual(ctx.reply, "ok")
        self.assertIn("write failed", ctx.compression_error)
        self.assertEqual(ctx.turns_since_compression, 1)
        self.assertEqual(ctx.model_messages()[1]["content"], LONG)

    async def test_many_steps_count_as_one_turn_and_preserve_tool_pairs(self):
        calls = [{"id": "one", "name": "unknown", "arguments": {}},
                 {"id": "two", "name": "unknown", "arguments": {}}]
        first = await self.turn(
            model=scripted({"tool_calls": calls}, {"content": ""}),
            compression_token_threshold=None, compression_turn_threshold=2)
        self.assertEqual(first.step_index, 2)
        self.assertEqual(first.turns_since_compression, 1)
        self.assertIsNone(first.reply)

        async def summary(*, messages, tools):
            ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
            self.assertEqual(ids, ["one", "two"])
            return {"content": "No order was executed; both tools were unknown."}

        second = await self.turn(compression_token_threshold=None,
                                 compression_turn_threshold=2, compression_model=summary)
        self.assertTrue(second.compression_applied)

    async def test_failed_turn_records_stats_without_running_summary(self):
        async def broken(*, messages, tools):
            raise RuntimeError("model unavailable")

        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            await self.turn(model=broken, force_compression=True, compression_model=scripted())
        rows = self.store.list_messages("s")
        self.assertEqual(rows[-1]["metadata"]["stop_reason"], "failed")
        self.assertFalse(any(m.get("name") == "compression_started" for m in rows))

    async def test_usage_does_not_trigger_token_threshold(self):
        ctx = await self.turn(
            model=scripted({"content": "ok", "usage": {"total_tokens": 10000000}}),
            compression_token_threshold=100000, compression_turn_threshold=None,
            count_tokens=counter, compression_model=scripted())
        self.assertFalse(ctx.compression_applied)
        self.assertLess(ctx.num_tokens, 100000)


class MemoryCompressionTests(CompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return MemoryStore()


class SQLiteCompressionTests(CompressionContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        store = SQLiteStore()
        store.initialize()
        self.addCleanup(store.close)
        return store

    async def test_reopening_database_restores_summary_and_round_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            store = SQLiteStore(path)
            try:
                store.initialize()
                store.create_session("s", "u")

                class Compressed(BaseAgent):
                    FORCE_COMPRESSION = True
                    COMPRESSION_MODEL = staticmethod(scripted({"content": "Summary"}))

                await Compressed.run_turn(
                    LONG, session_id="s", store=store, pipeline=[], model=scripted({"content": "ok"}))
            finally:
                store.close()
            reopened = SQLiteStore(path)
            try:

                class Uncompressed(BaseAgent):
                    COMPRESSION_TOKEN_THRESHOLD = None
                    COMPRESSION_TURN_THRESHOLD = None

                ctx = await Uncompressed.run_turn(
                    LONG, session_id="s", store=reopened, pipeline=[],
                    model=scripted({"content": "new"}))
                self.assertEqual(ctx.model_messages()[1], Context.summary_message("Summary"))
                self.assertEqual(ctx.turns_since_compression, 1)
                self.assertGreater(ctx.num_tokens, 0)
            finally:
                reopened.close()
