import asyncio
import json
import unittest

from cagent import BaseAgent, Context, Tool, ToolError, execute_tool_call
from cagent.stores import MemoryStore


def call(call_id="c", name="menu", arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments if arguments is not None else {}}


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools, tool_choice=None):
        return next(iterator)

    return model


def menu(*, store, ctx, agent_cls):
    assert store.list_messages(ctx.session_id)[-1]["tool_calls"]
    return {"coffees": ["拿铁", "美式"]}


class CoffeeAgent(BaseAgent):
    @classmethod
    def get_tools(cls, ctx):
        return {"menu": Tool(menu, "查询菜单")}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.store.create_session("s", "u")

    async def run_agent(self, model, **kwargs):
        return await CoffeeAgent.run_turn(
            kwargs.pop("input_message", "hello"), session_id="s",
            store=self.store, pipeline=[], model=model, **kwargs)

    async def test_full_tool_turn(self):
        requests = []

        async def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            self.assertEqual(tools[0]["name"], "menu")
            if len(requests) == 1:
                return {"tool_calls": [call()]}
            self.assertEqual(json.loads(messages[-1]["content"])["coffees"], ["拿铁", "美式"])
            return {"content": "今天有拿铁和美式。"}

        ctx = await CoffeeAgent.run_turn("菜单", session_id="s", store=self.store,
                                         pipeline=[], model=model)
        self.assertEqual(ctx.reply, "今天有拿铁和美式。")
        self.assertEqual(ctx.step_index, 2)
        self.assertEqual(self.store.get_status("s"), "IDLE")
        self.assertEqual([m["role"] for m in self.store.list_messages("s") if m["role"] != "mark"],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(ctx.model_messages(), Context(
            "s", messages=self.store.list_messages("s"),
            system_prompt=ctx.system_prompt).model_messages())
        with self.assertRaises(TypeError):
            CoffeeAgent()

    async def test_tools_receive_agent_subclass_and_shared_turn_dependencies(self):
        received = []

        def identify(*, store, ctx, agent_cls):
            received.append((agent_cls, store, ctx))
            return agent_cls.__name__

        async def identify_async(*, store, ctx, agent_cls):
            await asyncio.sleep(0)
            return identify(store=store, ctx=ctx, agent_cls=agent_cls)

        class ParentAgent(BaseAgent):
            @classmethod
            def identify_bound(cls, *, store, ctx, agent_cls):
                assert cls is agent_cls
                return identify(store=store, ctx=ctx, agent_cls=agent_cls)

            @classmethod
            def get_tools(cls, ctx):
                return {"sync": Tool(identify), "async": Tool(identify_async),
                        "bound": Tool(cls.identify_bound)}

        class ChildAgent(ParentAgent):
            pass

        ctx = await ChildAgent.run_turn(
            "Who are you?", session_id="s", store=self.store, pipeline=[],
            model=scripted({"tool_calls": [call(name, name) for name in ("sync", "async", "bound")]},
                           {"content": "done"}))
        self.assertEqual(len(received), 3)
        for agent_cls, store, tool_ctx in received:
            self.assertIs(agent_cls, ChildAgent)
            self.assertIs(store, self.store)
            self.assertIs(tool_ctx, ctx)
        self.assertEqual([m["content"] for m in ctx.model_messages() if m["role"] == "tool"],
                         ["ChildAgent"] * 3)

    async def test_backlog_stays_outside_active_turn_and_next_turn_reloads_history(self):
        started, release = asyncio.Event(), asyncio.Event()
        batch = [{"sender_id": "a", "content": "hello"}]
        requests = []

        async def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            started.set()
            await release.wait()
            return {"content": ""}

        task = asyncio.create_task(self.run_agent(model, input_message=batch))
        await asyncio.wait_for(started.wait(), 2)
        batch[:] = [{"sender_id": "b", "content": "拿铁"}, {"sender_id": "c", "content": "美式"}]
        release.set()
        first = await task
        self.assertIsNone(first.reply)
        second = await self.run_agent(model, input_message=batch)
        self.assertNotEqual(first.turn_id, second.turn_id)
        self.assertEqual(json.loads(requests[0][1]["content"])["messages"][0]["sender_id"], "a")
        users = [m for m in requests[1] if m["role"] == "user"]
        self.assertEqual(len(users), 2)
        self.assertEqual(json.loads(users[-1]["content"])["messages"], batch)

    async def test_tool_dispatch_reports_expected_errors_as_text(self):
        async def order(*, size, store, ctx, agent_cls):
            if size != "large":
                raise ToolError("只剩大杯")
            return {"size": size, "session": ctx.session_id}

        registry = {"order": Tool(order)}
        ctx = Context("s")
        cases = [
            (call(name="missing"), "未知工具"),
            (call(name="order"), "参数不匹配"),
            (call(name="order", arguments={"size": "large", "ctx": {}}), "保留字段"),
            (call(name="order", arguments={"size": "large", "store": {}}), "保留字段"),
            (call(name="order", arguments={"size": "large", "agent_cls": "Other"}), "保留字段"),
            (call(name="order", arguments={"size": "small"}), "只剩大杯"),
        ]
        for tool_call, fragment in cases:
            with self.subTest(name=tool_call["name"], arguments=tool_call["arguments"]):
                result = await execute_tool_call(
                    tool_call, registry, self.store, ctx, agent_cls=CoffeeAgent)
                self.assertIn(fragment, result["content"])
                self.assertEqual(result["type"], "extra")
        result = await execute_tool_call(call(name="order", arguments={"size": "large"}),
                                         registry, self.store, ctx, agent_cls=CoffeeAgent)
        self.assertEqual(json.loads(result["content"]), {"size": "large", "session": "s"})

    async def test_executor_skips_calls_after_stop_but_keeps_pairing(self):
        executed = []

        def act(*, store, ctx, agent_cls):
            executed.append(True)
            ctx.stop("completed")
            return "done"

        tools = {"act": Tool(act)}
        ctx = Context("s")
        first = await execute_tool_call(call("a", "act"), tools, self.store, ctx, agent_cls=BaseAgent)
        second = await execute_tool_call(call("b", "act"), tools, self.store, ctx, agent_cls=BaseAgent)
        self.assertEqual(executed, [True])
        self.assertEqual(first["content"], "done")
        self.assertEqual(first["type"], "extra")
        self.assertIn("未执行", second["content"])
        self.assertEqual(second["tool_call_id"], "b")
        self.assertEqual(second["type"], "extra")

    async def test_typed_tool_error_stays_extra(self):
        async def rejected(*, store, ctx, agent_cls):
            raise ToolError("检索失败")

        result = await execute_tool_call(call(name="recall"), {"recall": Tool(rejected, result_type="memory")},
                                         self.store, Context("s"), agent_cls=BaseAgent)
        self.assertEqual(result["type"], "extra")
        self.assertIn("检索失败", result["content"])

    async def test_interrupt_during_model_skips_tools_and_hides_draft(self):
        for has_tools in (True, False):
            with self.subTest(has_tools=has_tools):
                async def model(*, messages, tools, tool_choice=None):
                    self.store.set_status("s", "INTERRUPTED")
                    return {"content": "draft", "tool_calls": [call()] if has_tools else []}

                ctx = await self.run_agent(model)
                self.assertEqual(ctx.stop_reason, "interrupted")
                self.assertIsNone(ctx.reply)
                self.assertEqual(self.store.get_status("s"), "INTERRUPTED")
                if has_tools:
                    self.assertIn("未执行", ctx.model_messages()[-1]["content"])
                ctx.model_messages()

    async def test_interrupt_finishes_current_tool_and_skips_remaining(self):
        started, release = asyncio.Event(), asyncio.Event()
        executed = []

        async def slow(*, number, store, ctx, agent_cls):
            started.set()
            await release.wait()
            executed.append(number)
            return {"done": number}

        class SlowAgent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"slow": Tool(slow)}

        task = asyncio.create_task(SlowAgent.run_turn(
            "order", session_id="s", store=self.store, pipeline=[], model=scripted({"tool_calls": [
                call("first", "slow", {"number": 1}), call("second", "slow", {"number": 2})]})))
        await asyncio.wait_for(started.wait(), 2)
        self.store.set_status("s", "INTERRUPTED")
        release.set()
        ctx = await task
        self.assertEqual(executed, [1])
        self.assertEqual(ctx.stop_reason, "interrupted")
        self.assertIn("未执行", ctx.model_messages()[-1]["content"])
        ctx.model_messages()

    async def test_unexpected_tool_failure_is_not_retried(self):
        def broken(*, store, ctx, agent_cls):
            raise RuntimeError("connection lost after action")

        class BrokenAgent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"broken": Tool(broken)}

        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            await BrokenAgent.run_turn("order", session_id="s", store=self.store, pipeline=[],
                                       model=scripted({"tool_calls": [call(name="broken")]}))
        self.assertEqual(self.store.get_status("s"), "FAILED")
        self.assertEqual(self.store.list_messages("s")[-2]["role"], "assistant")
        self.assertEqual(self.store.list_messages("s")[-1]["metadata"]["stop_reason"], "failed")
        with self.assertRaisesRegex(ValueError, "recovery"):
            await self.run_agent(scripted({"content": "unsafe retry"}))
        self.assertEqual(len(self.store.list_messages("s")), 3)

    async def test_step_limit_preserves_tool_pairing(self):
        class LimitedAgent(CoffeeAgent):
            MAX_STEPS = 1

        ctx = await LimitedAgent.run_turn("hello", session_id="s", store=self.store, pipeline=[],
                                          model=scripted({"tool_calls": [call()]}))
        self.assertEqual(ctx.stop_reason, "step_limit")
        self.assertIsNone(ctx.reply)
        self.assertEqual(ctx.model_messages()[-1]["tool_call_id"], "c")
        ctx.model_messages()

    async def test_reused_call_id_is_rejected_before_tool_execution(self):
        await self.run_agent(scripted({"tool_calls": [call()]}, {"content": "done"}))
        with self.assertRaisesRegex(ValueError, "reused"):
            await self.run_agent(scripted({"tool_calls": [call()]}))
        self.assertEqual(sum(m["role"] == "tool" for m in self.store.list_messages("s")), 1)
        self.assertEqual(self.store.get_status("s"), "FAILED")

    async def test_concurrent_sessions_do_not_share_context(self):
        self.store.create_session("other", "u")
        arrived, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def model(*, messages, tools, tool_choice=None):
            calls.append(messages)
            if len(calls) == 2:
                arrived.set()
            await release.wait()
            return {"content": messages[-1]["content"]}

        tasks = [asyncio.create_task(BaseAgent.run_turn(
            sid, session_id=sid, store=self.store, pipeline=[], model=model)) for sid in ("s", "other")]
        await asyncio.wait_for(arrived.wait(), 2)
        release.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual([ctx.reply for ctx in results], ["s", "other"])
        self.assertNotEqual(results[0].turn_id, results[1].turn_id)

    async def test_cancellation_updates_status_and_propagates(self):
        started = asyncio.Event()

        async def waiting(*, messages, tools, tool_choice=None):
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(self.run_agent(waiting))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")
        result = await self.run_agent(scripted({"content": "next"}))
        self.assertEqual(result.reply, "next")
