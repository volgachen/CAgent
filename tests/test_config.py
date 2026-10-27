import asyncio
import unittest

from cagent import BaseAgent, Context, Tool
from cagent.stores import MemoryStore


async def finish(*, content, store, ctx, agent_cls):
    ctx.reply_content = content
    ctx.stop("completed")
    return f"【{ctx.agent_name}回复】{content}"


FINISH_TOOL = Tool(
    handler=finish, result_type="signal", description="完成本轮",
    parameters={"type": "object", "properties": {"content": {"type": "string"}},
                "required": ["content"], "additionalProperties": False},
)


class ConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_continues_after_text_and_finish_tool_provides_reply(self):
        store = MemoryStore()
        store.create_session("s", "u")
        requests = []
        events = []
        pipeline = ["Analyze first", "Then speak"]
        tools = {"finish": FINISH_TOOL}

        def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            if len(requests) == 1:
                return {"content": "Intermediate analysis"}
            return {"tool_calls": [{"id": "say", "name": "finish", "arguments": {"content": "Hello"}}]}

        class Writer(BaseAgent):
            NAME = "Writer"

        ctx = await Writer.run_turn("Hi", session_id="s", store=store, model=model,
                                    pipeline=pipeline, tools=tools, on_event=events.append)
        self.assertEqual(ctx.step_index, 2)
        self.assertEqual(ctx.stop_reason, "completed")
        self.assertEqual(ctx.reply, "Hello")
        self.assertEqual(requests[1][-1]["content"], "Then speak")
        self.assertEqual(pipeline, ["Analyze first", "Then speak"])
        self.assertEqual(list(tools), ["finish"])
        self.assertEqual([e["event"] for e in events], ["step_start", "step_end"] * 2)
        self.assertIn("Writer", ctx.model_messages()[-1]["content"])
        for request in requests:
            self.assertEqual(request[0], {"role": "system", "content": BaseAgent.SYSTEM_PROMPT})
            self.assertEqual(sum(m["role"] == "system" for m in request), 1)
        self.assertEqual([e["messages"] for e in events if e["event"] == "step_start"], requests)

    async def test_finish_stops_remaining_calls_but_keeps_tool_pairs(self):
        executed = []

        def action(*, store, ctx, agent_cls):
            executed.append(True)

        store = MemoryStore()
        store.create_session("s", "u")
        ctx = await BaseAgent.run_turn(
            "Hi", session_id="s", store=store, pipeline=[],
            tools={"finish": FINISH_TOOL, "action": Tool(action)},
            model=lambda **kw: {"tool_calls": [
                {"id": "c", "name": "finish", "arguments": {"content": "Hello"}},
                {"id": "a", "name": "action", "arguments": {}},
            ]})
        self.assertEqual(executed, [])
        self.assertEqual(ctx.reply, "Hello")
        self.assertIn("未执行", ctx.model_messages()[-1]["content"])
        self.assertEqual(ctx.model_messages()[-1]["tool_call_id"], "a")

    async def test_parallel_sessions_keep_tools_names_and_pipeline_separate(self):
        store = MemoryStore()
        for name in ("a", "b"):
            store.create_session(name, "u")
        arrived = 0
        ready = asyncio.Event()

        async def run(name):
            async def model(*, messages, tools, tool_choice=None):
                nonlocal arrived
                self.assertEqual([tool["name"] for tool in tools], ["finish"])
                self.assertEqual(messages[-1]["content"], f"Instruction {name}")
                self.assertEqual(messages[0], {"role": "system", "content": f"Persona {name}"})
                arrived += 1
                if arrived == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 2)
                return {"tool_calls": [{"id": name, "name": "finish",
                                        "arguments": {"content": name}}]}

            class NamedAgent(BaseAgent):
                NAME = name

            return await NamedAgent.run_turn(
                "Hi", session_id=name, store=store, model=model,
                tools={"finish": FINISH_TOOL}, pipeline=[f"Instruction {name}"],
                system_prompt=f"Persona {name}")

        contexts = await asyncio.gather(run("a"), run("b"))
        self.assertEqual([ctx.reply for ctx in contexts], ["a", "b"])
        self.assertEqual([ctx.agent_name for ctx in contexts], ["a", "b"])

    async def test_prompt_can_change_between_turns_without_persisting(self):
        store = MemoryStore()
        store.create_session("s", "u")
        requests = []

        def model(*, messages, tools, tool_choice=None):
            requests.append(messages)
            return {"content": "Hello"}

        for prompt in ("First persona", "Second persona", None):
            await BaseAgent.run_turn("Hi", session_id="s", store=store, model=model,
                                     pipeline=[], system_prompt=prompt)
        self.assertEqual([r[0]["content"] for r in requests],
                         ["First persona", "Second persona", BaseAgent.SYSTEM_PROMPT])
        self.assertTrue(all(sum(m["role"] == "system" for m in r) == 1 for r in requests))
        self.assertFalse(any(m["role"] == "system" for m in store.list_messages("s")))
        self.assertEqual(sum(m["role"] == "user" for m in requests[-1]), 3)

    async def test_invalid_prompt_fails_before_writing_turn(self):
        store = MemoryStore()
        store.create_session("s", "u")
        for prompt in ("", "  ", 123):
            with self.assertRaisesRegex(ValueError, "system_prompt"):
                await BaseAgent.run_turn("Hi", session_id="s", store=store,
                                         model=lambda **kw: self.fail("Model must not run"),
                                         system_prompt=prompt)
        self.assertEqual(store.list_messages("s"), [])
        self.assertEqual(store.get_status("s"), "IDLE")

    async def test_event_callback_failure_does_not_break_the_turn(self):
        store = MemoryStore()
        store.create_session("s", "u")

        def broken(event):
            raise RuntimeError("observer crashed")

        ctx = await BaseAgent.run_turn("Hi", session_id="s", store=store, pipeline=[],
                                       model=lambda **kw: {"content": "Hello"},
                                       on_event=broken)
        self.assertEqual(ctx.reply, "Hello")
        self.assertEqual(ctx.stop_reason, "completed")
