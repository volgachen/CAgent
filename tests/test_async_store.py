import asyncio
import unittest

from cagent import BaseAgent, Context, Tool
from cagent.stores import MemoryStore


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools):
        return next(iterator)

    return model


def call(call_id="c", name="act", arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments if arguments is not None else {}}


class AsyncStore:
    """Every protocol method is a coroutine, like a store over an async driver."""

    def __init__(self, inner):
        self.inner = inner

    async def create_session(self, session_id, user_id):
        return self.inner.create_session(session_id, user_id)

    async def get_session(self, session_id):
        return self.inner.get_session(session_id)

    async def get_status(self, session_id):
        await asyncio.sleep(0)
        return self.inner.get_status(session_id)

    async def set_status(self, session_id, status):
        return self.inner.set_status(session_id, status)

    async def list_messages(self, session_id, *, include_deleted=False):
        return self.inner.list_messages(session_id, include_deleted=include_deleted)

    async def save_messages(self, session_id, messages):
        await asyncio.sleep(0)
        return self.inner.save_messages(session_id, messages)

    async def delete_message(self, session_id, message_id):
        return self.inner.delete_message(session_id, message_id)


class MixedStore(MemoryStore):
    """Only save_messages is asynchronous; the rest stay synchronous."""

    async def save_messages(self, session_id, messages):
        return MemoryStore.save_messages(self, session_id, messages)


def make_agent(executed):
    def act(*, value, store, ctx, agent_cls):
        executed.append(value)
        return {"done": value}

    class ActingAgent(BaseAgent):
        @classmethod
        def get_tools(cls, ctx):
            return {"act": Tool(act, description="执行动作")}

    return ActingAgent


class AsyncStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        inner = MemoryStore()
        inner.create_session("s", "u")
        self.inner = inner
        self.store = AsyncStore(inner)

    async def test_full_tool_turn_runs_on_a_fully_async_store(self):
        executed = []
        agent = make_agent(executed)
        ctx = await agent.run_turn(
            "hi", session_id="s", store=self.store, pipeline=[],
            model=scripted({"tool_calls": [call(arguments={"value": 7})]}, {"content": "done"}))
        self.assertEqual(executed, [7])
        self.assertEqual(ctx.reply, "done")
        self.assertEqual(ctx.stop_reason, "completed")
        self.assertEqual(await self.store.get_status("s"), "IDLE")
        roles = [m["role"] for m in ctx.model_messages() if m["role"] != "system"]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertEqual(ctx.model_messages()[3]["tool_call_id"], "c")

    async def test_interrupt_is_observed_through_async_status_reads(self):
        async def model(*, messages, tools):
            await self.store.set_status("s", "INTERRUPTED")
            return {"content": "draft"}

        ctx = await BaseAgent.run_turn("hi", session_id="s", store=self.store,
                                       pipeline=[], model=model)
        self.assertEqual(ctx.stop_reason, "interrupted")
        self.assertIsNone(ctx.reply)
        self.assertEqual(await self.store.get_status("s"), "INTERRUPTED")

    async def test_forced_compression_runs_on_a_fully_async_store(self):
        class Compressing(BaseAgent):
            FORCE_COMPRESSION = True
            COMPRESSION_MODEL = staticmethod(scripted({"content": "Summary"},
                                                      {"content": "Summary 2"}))

        ctx = await Compressing.run_turn(
            "hello", session_id="s", store=self.store, pipeline=[],
            model=scripted({"content": "ok"}))
        self.assertTrue(ctx.compression_applied)
        self.assertIsNone(ctx.compression_error)
        self.assertEqual(ctx.model_messages(), [
            {"role": "system", "content": BaseAgent.SYSTEM_PROMPT},
            Context.summary_message("Summary")])

        second = await Compressing.run_turn(
            "again", session_id="s", store=self.store, pipeline=[],
            model=scripted({"content": "ok"}))
        self.assertTrue(second.compression_applied)
        self.assertEqual(second.model_messages()[1], Context.summary_message("Summary 2"))

    async def test_external_compression_runs_on_a_fully_async_store(self):
        await BaseAgent.run_turn("hello", session_id="s", store=self.store, pipeline=[],
                                 model=scripted({"content": "ok"}))
        result = await BaseAgent.compress_session(
            session_id="s", store=self.store, model=scripted({"content": "Summary"}))
        self.assertTrue(result.compression_applied)
        self.assertEqual(await self.store.get_status("s"), "IDLE")

    async def test_restore_running_on_async_store(self):
        ctx = Context("s")
        await self.store.set_status("s", "WAITING_CONFIRM")
        await BaseAgent.restore_running(self.store, ctx)
        self.assertEqual(await self.store.get_status("s"), "RUNNING")

    async def test_mixed_sync_and_async_methods(self):
        store = MixedStore()
        store.create_session("s", "u")
        executed = []
        agent = make_agent(executed)
        ctx = await agent.run_turn(
            "hi", session_id="s", store=store, pipeline=[],
            model=scripted({"tool_calls": [call(arguments={"value": 1})]}, {"content": "done"}))
        self.assertEqual(executed, [1])
        self.assertEqual(ctx.reply, "done")
        self.assertEqual(store.get_status("s"), "IDLE")
        self.assertEqual([m["role"] for m in store.list_messages("s") if m["role"] != "mark"],
                         ["user", "assistant", "tool", "assistant"])
