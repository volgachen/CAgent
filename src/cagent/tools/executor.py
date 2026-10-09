"""Registered tool dispatch; results and recoverable errors are plain text."""

import inspect
import json
from copy import deepcopy

from ..storage import Store, maybe_await
from .base import ToolError, ToolResult


async def execute_tool_call(tool_call, tools, store: Store, ctx, *, agent_cls):
    """Tools receive model arguments plus keyword-only store, ctx and agent_cls.

    Unknown tools, argument mismatches and ToolError return formatted text for
    the next model step; unexpected exceptions propagate unchanged. A turn that
    already stopped (completion or interruption) skips the call but still
    returns a paired result, keeping the tool history valid for the next model
    call. Every call first passes the agent's approve_tool_call hook; a denial
    returns a paired result instead of executing, and interrupts are re-checked
    after the hook because an approval may have waited for a human decision.
    """
    def message(value, result_type="extra"):
        return {"role": "tool", "type": result_type, "tool_call_id": tool_call["id"],
                "content": value if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, allow_nan=False)}

    async def check_interrupted():
        if not ctx.stop_flag and await maybe_await(store.get_status(ctx.session_id)) == "INTERRUPTED":
            ctx.stop("interrupted")

    def stopped():
        return message(f"本轮已停止，未执行工具 {tool_call['name']}")

    await check_interrupted()
    if ctx.stop_flag:
        return stopped()
    tool = tools.get(tool_call["name"])
    if tool is None:
        return message(f"工具执行失败：未知工具 {tool_call['name']}")
    decision = await agent_cls.approve_tool_call(store, ctx, tool_call, tool)
    await check_interrupted()
    if ctx.stop_flag:
        return stopped()
    if decision is not None and decision is not True:
        if isinstance(decision, str) and decision.strip():
            return message(f"工具调用未执行：{tool_call['name']} 未获批准（{decision.strip()}）")
        return message(f"工具调用未执行：{tool_call['name']} 未获批准")
    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict) or {"store", "ctx", "agent_cls"} & arguments.keys():
        return message(f"工具执行失败：{tool_call['name']} 参数必须是普通对象，且不能占用保留字段")
    try:
        inspect.signature(tool.handler).bind(
            **arguments, store=store, ctx=ctx, agent_cls=agent_cls)
    except TypeError as exc:
        return message(f"工具执行失败：{tool_call['name']} 参数不匹配（{exc}）")
    try:
        value = tool.handler(**arguments, store=store, ctx=ctx, agent_cls=agent_cls)
        if inspect.isawaitable(value):
            value = await value
    except ToolError as exc:
        return message(f"工具执行失败：{exc}")
    if isinstance(value, ToolResult):
        metadata = deepcopy(value.metadata)
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        return dict(message(value.content, tool.result_type), metadata=metadata)
    return message(value, tool.result_type)
