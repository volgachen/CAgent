import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from cagent.models import ModelError, make_openai_model


class ModelAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_encodes_tools_history_and_normalizes_tool_calls(self):
        call = {"id": "analysis", "name": "clock", "arguments": {}}
        messages = [{"role": "system", "content": "p"},
                    {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "content": "12:00", "tool_call_id": "analysis"}]
        reply = {"choices": [{"message": {"content": None, "tool_calls": [
            {"type": "function", "id": "say", "function": {
                "name": "finish", "arguments": '{"content":"hello"}'}}]}}]}
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m")
        definition = {"name": "finish", "description": "d", "parameters": {}}
        with patch("cagent.models.urlopen", return_value=io.StringIO(json.dumps(reply))) as send:
            output = await model(messages=messages, tools=[definition])
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(send.call_args.args[0].full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(payload["tools"], [{"type": "function", "function": definition}])
        self.assertEqual(json.loads(payload["messages"][1]["tool_calls"][0]["function"]["arguments"]), {})
        self.assertEqual(payload["messages"][2]["tool_call_id"], "analysis")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "p"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(output["tool_calls"], [{"id": "say", "name": "finish",
                                                 "arguments": {"content": "hello"}}])
        self.assertEqual(messages[1]["tool_calls"], [call])

    async def test_usage_and_refusal_fallback(self):
        response = {"choices": [{"message": {"content": None, "refusal": "抱歉"}}],
                    "usage": {"total_tokens": 12}}
        model = make_openai_model(base_url="https://example.invalid/v1/", model_name="test-model",
                                  api_key="test-key")
        with patch("cagent.models.urlopen", return_value=io.StringIO(json.dumps(response))) as send:
            result = await model(messages=[{"role": "user", "content": "你好"}], tools=[])
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(result, {"content": "抱歉", "tool_calls": [], "usage": {"total_tokens": 12}})

    async def test_reasoning_effort_is_sent_only_when_configured(self):
        response = {"choices": [{"message": {"content": "hi"}}]}
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m",
                                  reasoning_effort="low")
        with patch("cagent.models.urlopen", return_value=io.StringIO(json.dumps(response))) as send:
            await model(messages=[], tools=[])
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "low")

    async def test_http_errors_are_explicit_and_do_not_leak_bodies(self):
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m",
                                  api_key="secret")
        error = HTTPError("https://example.invalid", 401, "unauthorized", {},
                          io.BytesIO(b"sensitive response body"))
        with patch("cagent.models.urlopen", side_effect=error):
            with self.assertRaisesRegex(ModelError, "HTTP 401") as caught:
                await model(messages=[], tools=[])
        self.assertNotIn("sensitive", str(caught.exception))

    async def test_invalid_responses_raise_model_error(self):
        model = make_openai_model(base_url="https://example.invalid/v1", model_name="m")
        cases = [
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": {"content": "x", "tool_calls": [
                {"type": "function", "id": "i", "function": {
                    "name": "t", "arguments": "not-json"}}]}}]},
            {"choices": [{"message": {"content": "x", "tool_calls": [
                {"type": "builtin", "id": "i", "function": {
                    "name": "t", "arguments": "{}"}}]}}]},
        ]
        for response in cases:
            with self.subTest(response=response):
                with patch("cagent.models.urlopen",
                           return_value=io.StringIO(json.dumps(response))):
                    with self.assertRaisesRegex(ModelError, "没有返回有效"):
                        await model(messages=[], tools=[])
