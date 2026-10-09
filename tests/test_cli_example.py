import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cagent.stores import MemoryStore
from examples.cli import Assistant, chat_loop, mock_model


class CliLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_multi_turn_history_without_pipeline_instructions(self):
        store = MemoryStore()
        output = io.StringIO()
        with patch("builtins.input", side_effect=["我喜欢拿铁", "我上一句说了什么？", "/exit"]):
            with contextlib.redirect_stdout(output):
                await chat_loop(store, mock_model, "s")
        text = output.getvalue()
        self.assertIn("收到：我喜欢拿铁", text)
        self.assertIn("你上一句说的是：我喜欢拿铁", text)
        self.assertEqual(store.get_status("s"), "IDLE")

    async def test_tool_loop_finishes_with_tool_result(self):
        store = MemoryStore()
        output = io.StringIO()
        with patch("builtins.input", side_effect=["现在几点了", "/exit"]):
            with contextlib.redirect_stdout(output):
                await chat_loop(store, mock_model, "s")
        self.assertIn("现在的时间是", output.getvalue())

    async def test_history_new_and_exit_commands(self):
        store = MemoryStore()
        output = io.StringIO()
        with patch("builtins.input", side_effect=["", "你好", "/history", "/new", "/exit"]):
            with contextlib.redirect_stdout(output):
                await chat_loop(store, mock_model, "original")
        text = output.getvalue()
        self.assertIn("user：你好", text)
        self.assertIn("已开始新会话", text)
        self.assertEqual(store.get_status("original"), "IDLE")

    async def test_registered_tools_and_free_pipeline(self):
        store = MemoryStore()
        store.create_session("s", "u")
        seen = []

        async def model(*, messages, tools):
            seen.append((messages, tools))
            return {"tool_calls": [{"id": "f", "name": "finish",
                                    "arguments": {"content": "done"}}]}

        ctx = await Assistant.run_turn("hi", session_id="s", store=store, model=model)
        self.assertEqual(ctx.reply, "done")
        self.assertEqual([t["name"] for t in seen[0][1]], ["finish", "clock"])
        self.assertEqual(ctx.step_index, 1)
        self.assertIn("【助手回复】done", ctx.model_messages()[-1]["content"])


class CliCommandTests(unittest.TestCase):
    def test_script_runs_directly_and_reopens_sqlite_history(self):
        script = Path(__file__).resolve().parents[1] / "examples" / "cli.py"
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-X", "utf8", str(script), "--mock", "--db",
                       str(Path(directory) / "chat.db"), "--session", "persisted"]
            first = subprocess.run(command, input="我喜欢拿铁\n/exit\n", text=True,
                                   encoding="utf-8", capture_output=True, cwd=directory, timeout=15)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(command, input="我上一句说了什么？\n/exit\n", text=True,
                                    encoding="utf-8", capture_output=True, cwd=directory, timeout=15)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("你上一句说的是：我喜欢拿铁", second.stdout)
