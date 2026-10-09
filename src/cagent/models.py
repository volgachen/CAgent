"""OpenAI-compatible Chat Completions adapter; the only place wire format lives."""

import asyncio
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelError(RuntimeError):
    """The endpoint could not be reached or returned an unusable response."""


def make_openai_model(*, base_url, model_name, api_key="", reasoning_effort=None, timeout=60):
    """Return an async model callable for any OpenAI-compatible endpoint.

    reasoning_effort is sent only when provided; the request runs in a worker
    thread so the event loop keeps serving tools and interrupts.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def request_model(messages, tools):
        wire_messages = []
        for message in messages:
            wire = dict(message)
            if message.get("tool_calls"):
                wire["tool_calls"] = [
                    {"id": call["id"], "type": "function", "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    }} for call in message["tool_calls"]
                ]
            wire_messages.append(wire)
        payload = {"model": model_name, "messages": wire_messages, "stream": False}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            # Never echo response bodies; they may contain credentials or prompts.
            raise ModelError(f"模型接口返回 HTTP {exc.code}，请检查地址、模型名和 API Key。") from None
        except (URLError, TimeoutError) as exc:
            raise ModelError(f"模型网络请求失败（{type(exc).__name__}）。") from None
        try:
            message = data["choices"][0]["message"]
            calls = []
            for call in message.get("tool_calls") or []:
                if call["type"] != "function":
                    raise ValueError("Only function tools are supported")
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                calls.append({"id": call["id"], "name": call["function"]["name"], "arguments": arguments})
            content = message.get("content") or message.get("refusal") or ""
            if not isinstance(content, str) or (not content and not calls):
                raise ValueError("Expected text or tool calls")
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ModelError("模型没有返回有效的文字或工具调用，请检查接口响应格式。") from None
        return {"content": content, "tool_calls": calls, "usage": data.get("usage")}

    async def model(*, messages, tools):
        return await asyncio.to_thread(request_model, messages, tools)

    return model
