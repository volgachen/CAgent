import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cagent import BaseAgent, Context, Image, Tool, ToolError, ToolResult, relocate_tool_images
from cagent.models import make_openai_model
from cagent.stores import MemoryStore, SQLiteStore


URL_A = "https://example.invalid/a.png"
URL_B = "https://example.invalid/b.png"
DATA = "data:image/png;base64,iVBORw0KGgo="


def recording(*outputs):
    iterator, calls = iter(outputs), []

    async def model(*, messages, tools, tool_choice=None):
        calls.append(messages)
        return next(iterator)

    return model, calls


def call(call_id, name="shot", arguments=None):
    return {"id": call_id, "name": name, "arguments": arguments or {}}


def tool_messages(messages):
    return [m for m in messages if m["role"] == "tool"]


def image_urls(content):
    if not isinstance(content, list):
        return []
    return [p["image_url"]["url"] for p in content if p["type"] == "image_url"]


def shot_tool(*refs):
    """Tool returning one ToolResult per call, cycling through the given image lists."""
    queue = list(refs)

    def shot(*, store, ctx, agent_cls):
        return ToolResult(content="captured", images=queue.pop(0))

    return Tool(shot, description="take a screenshot")


class ToolResultValidationTests(unittest.TestCase):
    def test_images_are_normalized_at_construction(self):
        result = ToolResult("x", images=[URL_A, Image(URL_B, detail="low"), {"ref": DATA}])
        self.assertEqual(result.images, [{"ref": URL_A}, {"ref": URL_B, "detail": "low"}, {"ref": DATA}])
        self.assertEqual(ToolResult("x", images=(URL_A,)).images, [{"ref": URL_A}])
        self.assertEqual(ToolResult("x").images, [])

    def test_invalid_images_and_reserved_metadata_are_rejected(self):
        for kwargs in ({"images": [""]}, {"images": [{"ref": URL_A, "detail": "huge"}]},
                       {"images": URL_A}, {"images": [{"ref": URL_A, "at": 1}]},
                       {"metadata": {"images": []}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ToolResult("x", **kwargs)

    def test_interleaved_content_is_split_into_text_and_placed_images(self):
        result = ToolResult(["Page 1:", Image(URL_A), " Page 2:", Image(URL_B, detail="low"), " end"])
        self.assertEqual(result.content, "Page 1: Page 2: end")
        self.assertEqual(result.images, [{"ref": URL_A, "at": 7},
                                         {"ref": URL_B, "detail": "low", "at": 15}])
        self.assertEqual(ToolResult((Image(URL_A),)).images, [{"ref": URL_A, "at": 0}])

    def test_invalid_interleaved_content_is_rejected(self):
        for content, kwargs in (([], {}), ([" "], {}), (["x", 1], {}),
                                (["x", {"ref": URL_A}], {}), (["x", Image("")], {}),
                                (["x", Image(URL_A)], {"images": [URL_B]})):
            with self.subTest(content=content), self.assertRaises(ValueError):
                ToolResult(content, **kwargs)
        with self.assertRaises(TypeError):
            ToolResult(None)


class ToolImageContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")

    async def run_agent(self, agent, model, text="look"):
        return await agent.run_turn(text, session_id="s", store=self.store, model=model, pipeline=[])

    async def test_interleaved_tool_result_keeps_order_for_the_model(self):
        def pages(*, store, ctx, agent_cls):
            return ToolResult(["Page 1:", Image(URL_A), "Page 2:", Image(URL_B, detail="low"), "done"],
                              metadata={"pages": 2})

        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"shot": Tool(pages)}

        model, calls = recording({"tool_calls": [call("c1")]}, {"content": "ok"})
        await self.run_agent(Agent, model)
        content = tool_messages(calls[1])[0]["content"]
        self.assertEqual(content, [
            {"type": "text", "text": "Page 1:"},
            {"type": "image_url", "image_url": {"url": URL_A}},
            {"type": "text", "text": "Page 2:"},
            {"type": "image_url", "image_url": {"url": URL_B, "detail": "low"}},
            {"type": "text", "text": "done"},
        ])
        stored = [m for m in self.store.list_messages("s") if m["role"] == "tool"][0]
        self.assertEqual(stored["content"], "Page 1:Page 2:done")
        self.assertEqual(stored["metadata"], {"pages": 2, "images": [
            {"ref": URL_A, "at": 7}, {"ref": URL_B, "detail": "low", "at": 14}]})

    async def test_interleaved_tool_result_respects_the_limit(self):
        def pages(*, store, ctx, agent_cls):
            return ToolResult(["A:", Image(URL_A), "B:", Image(URL_B), "end"])

        class Agent(BaseAgent):
            TOOL_IMAGE_LIMIT = 1

            @classmethod
            def get_tools(cls, ctx):
                return {"shot": Tool(pages)}

        model, calls = recording({"tool_calls": [call("c1")]}, {"content": "ok"})
        await self.run_agent(Agent, model)
        content = tool_messages(calls[1])[0]["content"]
        self.assertEqual([p.get("text") or p["image_url"]["url"] for p in content],
                         ["A:B:", URL_B, "end" + Context.omitted_tool_images_note(1)])

    async def test_tool_images_reach_the_next_model_step(self):
        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"shot": shot_tool([URL_A, Image(URL_B, detail="low")])}

        model, calls = recording({"tool_calls": [call("c1")]}, {"content": "two images"})
        ctx = await self.run_agent(Agent, model)
        self.assertEqual(ctx.reply, "two images")
        result = tool_messages(calls[1])[0]
        self.assertEqual(result["tool_call_id"], "c1")
        self.assertEqual([p["type"] for p in result["content"]], ["text", "image_url", "image_url"])
        self.assertEqual(result["content"][0]["text"], "captured")
        self.assertEqual(image_urls(result["content"]), [URL_A, URL_B])
        self.assertEqual(result["content"][2]["image_url"]["detail"], "low")
        stored = [m for m in self.store.list_messages("s") if m["role"] == "tool"][0]
        self.assertEqual(stored["metadata"]["images"], [{"ref": URL_A}, {"ref": URL_B, "detail": "low"}])
        self.assertEqual(stored["content"], "captured")

    async def test_host_references_resolve_through_the_hook_once_per_turn(self):
        resolved = []
        tool = shot_tool(["file:1"], ["file:1"])

        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"shot": tool}

            @classmethod
            async def resolve_image(cls, store, ctx, image):
                resolved.append(image["ref"])
                return DATA

        model, calls = recording({"tool_calls": [call("c1")]}, {"tool_calls": [call("c2")]},
                                 {"content": "done"})
        await self.run_agent(Agent, model)
        self.assertEqual(image_urls(tool_messages(calls[2])[-1]["content"]), [DATA])
        self.assertEqual(resolved, ["file:1"])

    async def test_metadata_and_images_coexist(self):
        def shot(*, store, ctx, agent_cls):
            return ToolResult("captured", metadata={"page": 3}, images=[URL_A])

        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"shot": Tool(shot)}

        model, _ = recording({"tool_calls": [call("c1")]}, {"content": "ok"})
        await self.run_agent(Agent, model)
        stored = [m for m in self.store.list_messages("s") if m["role"] == "tool"][0]
        self.assertEqual(stored["metadata"], {"page": 3, "images": [{"ref": URL_A}]})

    async def test_errors_and_skipped_calls_carry_no_images(self):
        def broken(*, store, ctx, agent_cls):
            raise ToolError("no page")

        class Agent(BaseAgent):
            @classmethod
            def get_tools(cls, ctx):
                return {"shot": shot_tool([URL_A]), "broken": Tool(broken)}

            @classmethod
            async def approve_tool_call(cls, store, ctx, tool_call, tool):
                return "denied" if tool_call["id"] == "c2" else None

        model, calls = recording(
            {"tool_calls": [call("c1", "broken"), call("c2", "shot")]}, {"content": "ok"})
        await self.run_agent(Agent, model)
        for message in tool_messages(calls[1]):
            self.assertIsInstance(message["content"], str)
        for stored in [m for m in self.store.list_messages("s") if m["role"] == "tool"]:
            self.assertNotIn("images", stored["metadata"])

    async def test_tool_image_limit_keeps_the_newest_images(self):
        # get_tools runs every step, so the tool (and its queue) is built once.
        tool = shot_tool(["https://x/1a", "https://x/1b"],
                         ["https://x/2a", "https://x/2b"],
                         ["https://x/3a", "https://x/3b"])

        class Agent(BaseAgent):
            TOOL_IMAGE_LIMIT = 3

            @classmethod
            def get_tools(cls, ctx):
                return {"shot": tool}

        model, calls = recording(
            {"tool_calls": [call("c1")]}, {"tool_calls": [call("c2")]},
            {"tool_calls": [call("c3")]}, {"content": "done"})
        await self.run_agent(Agent, model, text=["compare", Image("https://x/user")])
        results = tool_messages(calls[3])
        self.assertIsInstance(results[0]["content"], str)
        self.assertIn("2 earlier tool image(s) omitted", results[0]["content"])
        self.assertEqual(image_urls(results[1]["content"]), ["https://x/2b"])
        self.assertIn("1 earlier tool image(s) omitted", results[1]["content"][0]["text"])
        self.assertEqual(image_urls(results[2]["content"]), ["https://x/3a", "https://x/3b"])
        user = [m for m in calls[3] if m["role"] == "user"][0]
        self.assertEqual(image_urls(user["content"]), ["https://x/user"])
        # Storage is untouched: every reference is still recorded.
        stored = [m for m in self.store.list_messages("s") if m["role"] == "tool"]
        self.assertEqual(sum(len(m["metadata"]["images"]) for m in stored), 6)

    async def test_zero_limit_drops_tool_images_but_keeps_user_images(self):
        class Agent(BaseAgent):
            TOOL_IMAGE_LIMIT = 0

            @classmethod
            def get_tools(cls, ctx):
                return {"shot": shot_tool([URL_A])}

        model, calls = recording({"tool_calls": [call("c1")]}, {"content": "ok"})
        await Agent.run_turn("look", session_id="s", store=self.store, model=model,
                             pipeline=[], images=[URL_B])
        self.assertEqual(tool_messages(calls[1])[0]["content"],
                         "captured" + Context.omitted_tool_images_note(1))
        self.assertEqual(image_urls([m for m in calls[1] if m["role"] == "user"][0]["content"]), [URL_B])

    async def test_invalid_limit_is_rejected_before_the_turn(self):
        for limit in (-1, 1.5, True, "3"):
            with self.subTest(limit=limit):
                class Agent(BaseAgent):
                    TOOL_IMAGE_LIMIT = limit

                with self.assertRaisesRegex(ValueError, "TOOL_IMAGE_LIMIT"):
                    await self.run_agent(Agent, recording({"content": "x"})[0])
        self.assertEqual(self.store.list_messages("s"), [])

    async def test_compression_respects_the_limit(self):
        summaries = []

        async def summarizer(*, messages, tools, tool_choice=None):
            summaries.append(messages)
            return {"content": "Summary of screenshots"}

        class Agent(BaseAgent):
            TOOL_IMAGE_LIMIT = 1
            FORCE_COMPRESSION = True
            COMPRESSION_MODEL = staticmethod(summarizer)

            @classmethod
            def get_tools(cls, ctx):
                return {"shot": shot_tool([URL_A, URL_B])}

        model, _ = recording({"tool_calls": [call("c1")]}, {"content": "ok"})
        ctx = await self.run_agent(Agent, model, text="look " * 200)
        self.assertTrue(ctx.compression_applied, ctx.compression_error)
        sent = [url for m in summaries[0] for url in image_urls(m["content"])]
        self.assertEqual(sent, [URL_B])


class MemoryToolImageTests(ToolImageContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return MemoryStore()


class SQLiteToolImageTests(ToolImageContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        store = SQLiteStore()
        store.initialize()
        self.addCleanup(store.close)
        return store

    async def test_tool_images_survive_reopening_the_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            store = SQLiteStore(path)
            try:
                store.initialize()
                store.create_session("s", "u")

                class Agent(BaseAgent):
                    @classmethod
                    def get_tools(cls, ctx):
                        def pages(*, store, ctx, agent_cls):
                            return ToolResult(["before ", Image(URL_A), " after"])
                        return {"shot": shot_tool([URL_A]), "pages": Tool(pages)}

                model, _ = recording({"tool_calls": [call("c1"), call("c2", "pages")]}, {"content": "ok"})
                await Agent.run_turn("look", session_id="s", store=store, model=model, pipeline=[])
            finally:
                store.close()
            reopened = SQLiteStore(path)
            try:
                model, calls = recording({"content": "still see it"})
                await BaseAgent.run_turn("again", session_id="s", store=reopened,
                                         model=model, pipeline=[])
                results = tool_messages(calls[0])
                self.assertEqual(image_urls(results[0]["content"]), [URL_A])
                self.assertEqual([p["type"] for p in results[1]["content"]],
                                 ["text", "image_url", "text"])
            finally:
                reopened.close()


class RelocateToolImagesTests(unittest.TestCase):
    def image(self, url):
        return {"type": "image_url", "image_url": {"url": url}}

    def test_images_move_after_each_contiguous_tool_run(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]},
            {"role": "tool", "tool_call_id": "a", "content": [
                {"type": "text", "text": "shot a"}, self.image(URL_A)]},
            {"role": "tool", "tool_call_id": "b", "content": "no image"},
            {"role": "assistant", "content": "", "tool_calls": [call("c")]},
            {"role": "tool", "tool_call_id": "c", "content": [
                {"type": "text", "text": "shot c"}, self.image(URL_B), self.image(DATA)]},
        ]
        original = json.dumps(messages)
        out = relocate_tool_images(messages)
        self.assertEqual(json.dumps(messages), original)  # input not mutated
        self.assertEqual([m["role"] for m in out],
                         ["user", "assistant", "tool", "tool", "user", "assistant", "tool", "user"])
        self.assertEqual(out[2]["content"], "shot a\n[1 image(s) attached in the following message]")
        self.assertEqual(out[3]["content"], "no image")
        first = out[4]["content"]
        self.assertIn("tool output", first[0]["text"])
        self.assertEqual(first[1], {"type": "text", "text": "[tool call a]"})
        self.assertEqual(image_urls(first), [URL_A])
        self.assertEqual(image_urls(out[7]["content"]), [URL_B, DATA])
        self.assertTrue(all(isinstance(m["content"], str) for m in out if m["role"] == "tool"))

    def test_messages_without_tool_images_are_unchanged(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, self.image(URL_A)]},
                    {"role": "assistant", "content": "", "tool_calls": [call("a")]},
                    {"role": "tool", "tool_call_id": "a", "content": "plain"}]
        self.assertEqual(relocate_tool_images(messages), messages)


class AdapterToolImageTests(unittest.IsolatedAsyncioTestCase):
    MESSAGES = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": [
            {"type": "text", "text": "shot"}, {"type": "image_url", "image_url": {"url": DATA}}]},
    ]

    async def send(self, **options):
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m", **options)
        reply = {"choices": [{"message": {"content": "ok"}}]}
        with patch("cagent.models.urlopen", return_value=io.StringIO(json.dumps(reply))) as send:
            await model(messages=self.MESSAGES, tools=[])
        return json.loads(send.call_args.args[0].data)["messages"]

    async def test_inline_is_the_default(self):
        wire = await self.send()
        self.assertEqual(wire[2]["content"], self.MESSAGES[2]["content"])
        self.assertEqual(len(wire), 3)

    async def test_user_message_mode_relocates_images(self):
        wire = await self.send(tool_images="user_message")
        self.assertIsInstance(wire[2]["content"], str)
        self.assertEqual(wire[3]["role"], "user")
        self.assertEqual(image_urls(wire[3]["content"]), [DATA])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_openai_model(base_url="https://x", model_name="m", tool_images="sideways")


if __name__ == "__main__":
    unittest.main()
