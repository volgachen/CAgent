import asyncio
import unittest

from cagent import BaseAgent, Context, Tool
from cagent.stores import MemoryStore


def scripted(*outputs):
    iterator = iter(outputs)

    async def model(*, messages, tools, tool_choice=None):
        return next(iterator)

    return model


def call(call_id="c", name="act", arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments if arguments is not None else {}}


class ApprovalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.store.create_session("s", "u")
        self.executed = []

    def act_tool(self):
        executed = self.executed

        def act(*, store, ctx, agent_cls):
            executed.append(True)
            return "ran"

        return Tool(act, description="执行动作")

    async def test_default_hook_executes_tools(self):
        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

        ctx = await Agent.run_turn("hi", session_id="s", store=self.store, pipeline=[],
                                   model=scripted({"tool_calls": [call()]}, {"content": "done"}))
        self.assertEqual(self.executed, [True])
        self.assertEqual(ctx.reply, "done")

    async def test_denial_skips_execution_and_reports_reason_to_model(self):
        requests = []

        async def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            if len(requests) == 1:
                return {"tool_calls": [call(arguments={"path": "/etc/passwd"})]}
            return {"content": "好的，我不执行了"}

        class Guarded(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                self.assertEqual(tool_call["id"], "c")
                self.assertEqual(tool.description, "执行动作")
                if tool_call["arguments"].get("path") == "/etc/passwd":
                    return "该路径不在允许的目录内"
                return None

        ctx = await Guarded.run_turn("hi", session_id="s", store=self.store, pipeline=[],
                                     model=model)
        self.assertEqual(self.executed, [])
        self.assertEqual(ctx.reply, "好的，我不执行了")
        results = [m for m in requests[1] if m["role"] == "tool"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool_call_id"], "c")
        self.assertIn("未获批准", results[0]["content"])
        self.assertIn("该路径不在允许的目录内", results[0]["content"])
        saved = [m for m in self.store.list_messages("s") if m["role"] == "tool"]
        self.assertEqual(saved[0]["type"], "extra")
        ctx.model_messages()

    async def test_false_or_blank_denials_use_generic_text(self):
        for decision in (False, "", "   "):
            with self.subTest(decision=decision):
                store = MemoryStore()
                store.create_session("s", "u")
                executed = []

                def act(*, store, ctx, agent_cls):
                    executed.append(True)
                    return "ran"

                class Guarded(BaseAgent):
                    @classmethod
                    def get_tools(cls, ctx):
                        return {"act": Tool(act, description="执行动作")}

                    @classmethod
                    async def approve_tool_call(cls, store, ctx, tool_call, tool):
                        return decision

                ctx = await Guarded.run_turn(
                    "hi", session_id="s", store=store, pipeline=[],
                    model=scripted({"tool_calls": [call()]}, {"content": "ok"}))
                self.assertEqual(executed, [])
                results = [m for m in ctx.model_messages() if m["role"] == "tool"]
                self.assertEqual(len(results), 1)
                self.assertIn("未获批准", results[0]["content"])

    async def test_true_allows_execution(self):
        class Guarded(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                return True

        ctx = await Guarded.run_turn("hi", session_id="s", store=self.store, pipeline=[],
                                     model=scripted({"tool_calls": [call()]}, {"content": "done"}))
        self.assertEqual(self.executed, [True])
        self.assertEqual(ctx.reply, "done")

    async def test_hook_can_await_external_decision_with_waiting_status(self):
        arrived, release = asyncio.Event(), asyncio.Event()

        class Guarded(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                store.set_status(ctx.session_id, "WAITING_CONFIRM")
                arrived.set()
                await release.wait()
                await cls.restore_running(store, ctx)
                return None

        task = asyncio.create_task(Guarded.run_turn(
            "hi", session_id="s", store=self.store, pipeline=[],
            model=scripted({"tool_calls": [call()]}, {"content": "done"})))
        await asyncio.wait_for(arrived.wait(), 2)
        self.assertEqual(self.store.get_status("s"), "WAITING_CONFIRM")
        release.set()
        ctx = await task
        self.assertEqual(self.executed, [True])
        self.assertEqual(ctx.reply, "done")
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_interrupt_during_approval_wait_skips_execution(self):
        arrived, release = asyncio.Event(), asyncio.Event()

        class Guarded(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                arrived.set()
                await release.wait()
                return None  # approved, but the platform interrupted meanwhile

        task = asyncio.create_task(Guarded.run_turn(
            "hi", session_id="s", store=self.store, pipeline=[],
            model=scripted({"tool_calls": [call()]})))
        await asyncio.wait_for(arrived.wait(), 2)
        self.store.set_status("s", "INTERRUPTED")
        release.set()
        ctx = await task
        self.assertEqual(self.executed, [])
        self.assertEqual(ctx.stop_reason, "interrupted")
        self.assertIn("未执行", ctx.model_messages()[-1]["content"])
        self.assertEqual(ctx.model_messages()[-1]["tool_call_id"], "c")
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")

    async def test_restore_running_respects_interrupted_status(self):
        ctx = Context("s")
        self.store.set_status("s", "WAITING_CONFIRM")
        await BaseAgent.restore_running(self.store, ctx)
        self.assertEqual(self.store.get_status("s"), "RUNNING")
        self.store.set_status("s", "INTERRUPTED")
        await BaseAgent.restore_running(self.store, ctx)
        self.assertEqual(self.store.get_status("s"), "INTERRUPTED")

    async def test_denial_can_queue_supplementary_instruction(self):
        requests = []

        async def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            if len(requests) == 1:
                return {"tool_calls": [call(arguments={"target": "x"})]}
            return {"content": "已按补充说明改用只读方式"}

        class Guarded(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"act": self.act_tool()}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                ctx.pending_messages.append("用户补充：请改用只读方式完成")
                return "默认路径被拒绝"

        ctx = await Guarded.run_turn("hi", session_id="s", store=self.store, pipeline=[],
                                     model=model)
        self.assertEqual(self.executed, [])
        self.assertEqual(ctx.reply, "已按补充说明改用只读方式")
        second = requests[1]
        self.assertEqual(second[-1]["role"], "user")
        self.assertEqual(second[-1]["content"], "用户补充：请改用只读方式完成")
        denial = [m for m in second if m["role"] == "tool"][0]
        self.assertIn("默认路径被拒绝", denial["content"])

    async def test_unknown_tool_returns_error_without_asking_approval(self):
        asked = []

        class Guarded(BaseAgent):
            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                asked.append(tool_call["name"])
                return None

        ctx = await Guarded.run_turn("hi", session_id="s", store=self.store, pipeline=[],
                                     model=scripted({"tool_calls": [call(name="missing")]},
                                                    {"content": "ok"}))
        self.assertEqual(asked, [])
        results = [m for m in ctx.model_messages() if m["role"] == "tool"]
        self.assertIn("未知工具", results[0]["content"])
