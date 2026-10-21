import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cagent import BaseAgent, Context, Image, Tool
from cagent.models import make_openai_model
from cagent.stores import MemoryStore, SQLiteStore


URL = "https://example.invalid/cat.png"
DATA = "data:image/png;base64,iVBORw0KGgo="


def recording(*outputs):
    """Scripted model that records the messages of every call."""
    iterator, calls = iter(outputs), []

    async def model(*, messages, tools):
        calls.append(messages)
        return next(iterator)

    return model, calls


def call(call_id="c", name="act"):
    return {"id": call_id, "name": name, "arguments": {}}


class FileIdAgent(BaseAgent):
    """Host agent whose store keeps file IDs; resolution returns a data URL."""

    resolved = []

    @classmethod
    async def resolve_image(cls, store, ctx, image):
        cls.resolved.append(image)
        if image["ref"].startswith("file:"):
            return DATA
        return await super().resolve_image(store, ctx, image)


class ImageInputContract:
    def setUp(self):
        self.store = self.make_store()
        self.store.create_session("s", "u")
        FileIdAgent.resolved = []

    async def test_images_reach_model_as_image_url_parts(self):
        model, calls = recording({"content": "a cat"})
        ctx = await BaseAgent.run_turn("what is this?", session_id="s", store=self.store,
                                       model=model, pipeline=[],
                                       images=[URL, {"ref": DATA, "detail": "low"}])
        self.assertEqual(ctx.reply, "a cat")
        self.assertEqual(calls[0][-1], {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": URL}},
            {"type": "image_url", "image_url": {"url": DATA, "detail": "low"}},
        ]})

    async def test_only_references_are_persisted(self):
        model, _ = recording({"content": "ok"})
        await FileIdAgent.run_turn("look", session_id="s", store=self.store, model=model,
                                   pipeline=[], images=["file:42"])
        signal = next(m for m in self.store.list_messages("s") if m["type"] == "signal")
        self.assertEqual(signal["content"], "look")
        self.assertEqual(signal["metadata"], {"images": [{"ref": "file:42"}]})
        self.assertNotIn(DATA, json.dumps(self.store.list_messages("s")))

    async def test_resolved_once_per_turn_and_again_in_later_turns(self):
        model, calls = recording({"tool_calls": [call()]}, {"content": "done"}, {"content": "again"})
        tools = {"act": Tool(lambda *, store, ctx, agent_cls: "acted")}
        await FileIdAgent.run_turn("look", session_id="s", store=self.store, model=model,
                                   tools=tools, pipeline=[], images=["file:1"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(FileIdAgent.resolved, [{"ref": "file:1"}])
        # History keeps the image on later turns; a new turn resolves it afresh.
        await FileIdAgent.run_turn("and now?", session_id="s", store=self.store,
                                   model=model, pipeline=[])
        self.assertEqual(calls[2][1]["content"][1], {"type": "image_url", "image_url": {"url": DATA}})
        self.assertEqual(len(FileIdAgent.resolved), 2)

    async def test_image_only_input_is_allowed(self):
        model, calls = recording({"content": "a cat"})
        await BaseAgent.run_turn("", session_id="s", store=self.store, model=model,
                                 pipeline=[], images=[URL])
        self.assertEqual(calls[0][-1]["content"], [{"type": "image_url", "image_url": {"url": URL}}])

    async def test_invalid_images_are_rejected_before_any_write(self):
        model, _ = recording()
        for images in ("cat.png", [""], [{"ref": URL, "detail": "max"}],
                       [{"ref": URL, "extra": 1}], [3]):
            with self.subTest(images=images), self.assertRaises(ValueError):
                await BaseAgent.run_turn("x", session_id="s", store=self.store,
                                         model=model, pipeline=[], images=images)
        with self.assertRaises(ValueError):
            await BaseAgent.run_turn("  ", session_id="s", store=self.store, model=model, pipeline=[])
        self.assertEqual(self.store.list_messages("s"), [])
        self.assertEqual(self.store.get_status("s"), "IDLE")

    async def test_unresolvable_reference_fails_the_turn(self):
        model, calls = recording({"content": "unused"})
        with self.assertRaisesRegex(ValueError, "override resolve_image"):
            await BaseAgent.run_turn("x", session_id="s", store=self.store, model=model,
                                     pipeline=[], images=["file:9"])
        self.assertEqual(calls, [])
        self.assertEqual(self.store.get_status("s"), "FAILED")

    async def test_compression_model_receives_images(self):
        summaries = []

        async def summarizer(*, messages, tools):
            summaries.append(messages)
            return {"content": "User showed a cat photo; assistant identified it."}

        configured = type("Compressing", (FileIdAgent,), {
            "FORCE_COMPRESSION": True, "COMPRESSION_MODEL": staticmethod(summarizer)})
        model, _ = recording({"content": "a cat " * 200})
        ctx = await configured.run_turn("what is this?", session_id="s", store=self.store,
                                        model=model, pipeline=[], images=["file:7"])
        self.assertIsNone(ctx.compression_error)
        self.assertTrue(ctx.compression_applied)
        self.assertEqual(summaries[0][1]["content"][1], {"type": "image_url", "image_url": {"url": DATA}})
        # Resolved once for the turn; the compression call reused the turn cache.
        self.assertEqual(FileIdAgent.resolved, [{"ref": "file:7"}])
        # After compression the summary text replaces the image history.
        nxt, calls = recording({"content": "ok"})
        await configured.run_turn("thanks", session_id="s", store=self.store, model=nxt, pipeline=[])
        self.assertNotIn("image_url", json.dumps(calls[0]))

    async def test_external_compression_resolves_images(self):
        model, _ = recording({"content": "a cat " * 200})
        await FileIdAgent.run_turn("what?", session_id="s", store=self.store, model=model,
                                   pipeline=[], images=["file:3"])
        seen = []

        async def summarizer(*, messages, tools):
            seen.append(messages)
            return {"content": "Cat photo discussed."}

        ctx = await FileIdAgent.compress_session(session_id="s", store=self.store, model=summarizer)
        self.assertTrue(ctx.compression_applied, ctx.compression_error)
        self.assertIn({"type": "image_url", "image_url": {"url": DATA}}, seen[0][1]["content"])


    async def test_multiple_images_keep_list_order(self):
        model, calls = recording({"content": "ok"})
        await FileIdAgent.run_turn("compare", session_id="s", store=self.store, model=model,
                                   pipeline=[], images=["file:a", URL, Image("file:b", "high")])
        self.assertEqual([p.get("image_url") for p in calls[0][-1]["content"]], [
            None, {"url": DATA}, {"url": URL}, {"url": DATA, "detail": "high"}])
        self.assertEqual(FileIdAgent.resolved, [{"ref": "file:a"}, {"ref": URL}, {"ref": "file:b", "detail": "high"}])

    async def test_interleaved_input_keeps_text_and_image_order(self):
        model, calls = recording({"content": "ok"})
        await FileIdAgent.run_turn(
            ["Old version: ", Image("file:a"), " New version: ", Image(URL, "low"), " Which is better?"],
            session_id="s", store=self.store, model=model, pipeline=[])
        self.assertEqual(calls[0][-1]["content"], [
            {"type": "text", "text": "Old version: "},
            {"type": "image_url", "image_url": {"url": DATA}},
            {"type": "text", "text": " New version: "},
            {"type": "image_url", "image_url": {"url": URL, "detail": "low"}},
            {"type": "text", "text": " Which is better?"},
        ])
        signal = next(m for m in self.store.list_messages("s") if m["type"] == "signal")
        # Stored as plain text plus offsets, so text-only consumers still read it.
        self.assertEqual(signal["content"], "Old version:  New version:  Which is better?")
        self.assertEqual(signal["metadata"]["images"], [
            {"ref": "file:a", "at": 13}, {"ref": URL, "detail": "low", "at": 27}])

    async def test_interleaved_edges_and_survival_across_turns(self):
        model, calls = recording({"content": "ok"}, {"content": "ok"})
        await FileIdAgent.run_turn([Image("file:a"), Image("file:b"), "both?"], session_id="s",
                                   store=self.store, model=model, pipeline=[])
        await FileIdAgent.run_turn("again", session_id="s", store=self.store, model=model, pipeline=[])
        expected = [{"type": "image_url", "image_url": {"url": DATA}},
                    {"type": "image_url", "image_url": {"url": DATA}},
                    {"type": "text", "text": "both?"}]
        self.assertEqual(calls[0][-1]["content"], expected)
        self.assertEqual(calls[1][1]["content"], expected)

    async def test_invalid_interleaved_input_is_rejected(self):
        model, _ = recording()
        for message, kwargs in (([" ", "  "], {}), (["x", {"role": "user"}], {}),
                                (["x", Image("")], {}), (["x", Image(URL, "max")], {}),
                                (["x", Image(URL)], {"images": [URL]})):
            with self.subTest(message=message), self.assertRaises(ValueError):
                await BaseAgent.run_turn(message, session_id="s", store=self.store, model=model,
                                         pipeline=[], **kwargs)
        self.assertEqual(self.store.list_messages("s"), [])

    async def test_text_only_turn_is_unchanged(self):
        class Legacy(BaseAgent):
            @classmethod
            async def pre_run_turn(cls, store, session_id, input_message, max_steps):
                return await super().pre_run_turn(store, session_id, input_message, max_steps)

        model, calls = recording({"content": "ok"})
        await Legacy.run_turn("hi", session_id="s", store=self.store, model=model, pipeline=[])
        self.assertEqual(calls[0][-1], {"role": "user", "content": "hi"})
        self.assertEqual(self.store.list_messages("s")[0]["metadata"], {})

    async def test_message_batch_input_is_unchanged(self):
        model, calls = recording({"content": "ok"})
        await BaseAgent.run_turn([{"from": "a", "text": "hi"}], session_id="s", store=self.store,
                                 model=model, pipeline=[])
        self.assertEqual(calls[0][-1], {"role": "user", "content": json.dumps(
            {"messages": [{"from": "a", "text": "hi"}]}, ensure_ascii=False)})


class MemoryImageTests(ImageInputContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        return MemoryStore()


class SQLiteImageTests(ImageInputContract, unittest.IsolatedAsyncioTestCase):
    def make_store(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        store = SQLiteStore(str(Path(self.tmp.name) / "s.db"))
        self.addCleanup(store.close)
        store.initialize()
        return store


class ProjectionTests(unittest.TestCase):
    def test_text_messages_keep_plain_string_content(self):
        ctx = Context("s", messages=[{"id": "1", "role": "user", "content": "hi", "metadata": {}}])
        self.assertEqual(ctx.model_messages(), [{"role": "user", "content": "hi"}])

    def test_images_project_to_references_not_urls(self):
        ctx = Context("s", messages=[{"id": "1", "role": "user", "content": "hi",
                                      "metadata": {"images": [{"ref": "file:1"}]}}])
        self.assertEqual(ctx.model_messages()[0]["content"][1], {"type": "image_ref", "ref": "file:1"})

    def test_corrupt_offsets_are_reported_as_history_errors(self):
        for images in ([{"ref": "a", "at": 9}], [{"ref": "a", "at": 2}, {"ref": "b", "at": 1}],
                       [{"ref": "a", "at": True}]):
            ctx = Context("s", messages=[{"id": "1", "role": "user", "content": "hi",
                                          "metadata": {"images": images}}])
            with self.subTest(images=images), self.assertRaises(ValueError):
                ctx.model_messages()


class AdapterImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_parts_pass_through_to_the_wire(self):
        parts = [{"type": "text", "text": "what?"},
                 {"type": "image_url", "image_url": {"url": URL, "detail": "high"}}]
        reply = {"choices": [{"message": {"content": "a cat"}}]}
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m")
        with patch("cagent.models.urlopen", return_value=io.StringIO(json.dumps(reply))) as send:
            output = await model(messages=[{"role": "user", "content": parts}], tools=[])
        self.assertEqual(json.loads(send.call_args.args[0].data)["messages"][0]["content"], parts)
        self.assertEqual(output["content"], "a cat")


if __name__ == "__main__":
    unittest.main()
