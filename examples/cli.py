"""CAgent 终端助手示例。直接运行本文件：--mock 离线，默认读取环境变量调用真实模型。

演示框架的最小用法：一个 finish 完成工具、一个真实业务工具（clock）、
自由模型/工具循环（pipeline=[]）、可选 SQLite 持久化与 /history 查看。
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Allow "python examples/cli.py" without installing the local package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cagent import BaseAgent, Context, Tool, ToolError, MemoryStore, SQLiteStore
from cagent.models import make_openai_model


async def finish(*, content: str, store, ctx, agent_cls):
    """content 为本轮最终回复；调用即视为本轮完成。"""
    if not isinstance(content, str) or not content.strip():
        raise ToolError("content 必须是非空的最终回复")
    ctx.reply_content = content
    ctx.stop("completed")
    return f"【{ctx.agent_name}回复】{content}"


FINISH_TOOL = Tool(
    handler=finish,
    result_type="signal",
    description="给出本轮的最终回复，并正常结束这一轮。",
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "最终回复内容。"},
        },
        "required": ["content"],
        "additionalProperties": False,
    },
)


async def clock(*, store, ctx, agent_cls):
    """一个最小真实工具：读取本机当前时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


CLOCK_TOOL = Tool(
    handler=clock,
    description="查询助手所在系统的当前本地时间。",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


class Assistant(BaseAgent):
    NAME = "助手"
    SYSTEM_PROMPT = "你是一个简洁友好的中文助手。需要时调用 clock 查询时间；每次回复前先想清楚是否已经有足够信息。"
    COMPRESSION_TOKEN_THRESHOLD = None
    COMPRESSION_TURN_THRESHOLD = None

    @classmethod
    def get_tools(cls, ctx):
        return {"finish": FINISH_TOOL, "clock": CLOCK_TOOL}


async def mock_model(*, messages, tools, tool_choice=None):
    """离线固定规则，只验证工具循环、完成约定和多轮历史，不模拟真实理解。"""
    if messages[-1]["role"] == "tool":
        reply = f"现在的时间是{messages[-1]['content']}。"
        return {"tool_calls": [{"id": uuid4().hex, "name": "finish", "arguments": {"content": reply}}]}
    users = [m["content"] for m in messages if m["role"] == "user"]
    text = users[-1]
    if "时间" in text or "几点" in text:
        return {"tool_calls": [{"id": uuid4().hex, "name": "clock", "arguments": {}}]}
    if "上一句" in text:
        reply = f"你上一句说的是：{users[-2]}" if len(users) > 1 else "这是你刚说的第一句话呀。"
    else:
        reply = f"收到：{text}"
    return {"tool_calls": [{"id": uuid4().hex, "name": "finish", "arguments": {"content": reply}}]}


def show_message(message):
    label = message["role"]
    if message.get("tool_call_id"):
        label += f" ({message['tool_call_id']})"
    print(f"  {label}：{message.get('content') or '（无文本）'}", flush=True)
    for call in message.get("tool_calls", []):
        print(f"  工具调用 {call['id']} → {call['name']}："
              f"{json.dumps(call['arguments'], ensure_ascii=False)}", flush=True)


def trace_event(event):
    step = event["step"]
    if event["event"] == "step_start":
        print(f"\n[Step {step}] 模型输入（含历史）", flush=True)
        for message in event["messages"]:
            show_message(message)
        print("  可用工具：" + (json.dumps(event["tools"], ensure_ascii=False) or "无"), flush=True)
    elif event["event"] == "step_end":
        print(f"[Step {step}] 本步已保存的模型输出与工具结果", flush=True)
        for message in event["messages"]:
            if message["role"] != "mark":
                show_message(message)


def ensure_session(store, session_id):
    try:
        store.get_session(session_id)
    except KeyError:
        store.create_session(session_id, "cli-user")


async def chat_loop(store, model, session_id, *, show_stats=False, show_trace=False):
    ensure_session(store, session_id)
    print(f"会话：{session_id}")
    print("输入文字开始对话；/history 查看历史，/new 新会话，/exit 退出。")
    while True:
        try:
            text = input("你：").strip()
        except EOFError:
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/new":
            session_id = "chat-" + uuid4().hex[:12]
            ensure_session(store, session_id)
            print(f"已开始新会话：{session_id}")
            continue
        if text == "/history":
            ctx = Context(session_id, messages=store.list_messages(session_id, include_deleted=True),
                          system_prompt=Assistant.SYSTEM_PROMPT)
            for message in ctx.model_messages():
                print(f"{message['role']}：{message['content']}")
            continue
        try:
            if show_trace:
                print("\n--- 本轮开始 ---", flush=True)
            ctx = await Assistant.run_turn(
                text, session_id=session_id, store=store, model=model,
                on_event=trace_event if show_trace else None)
        except Exception as exc:
            print(f"本轮失败：{type(exc).__name__}: {exc}")
            continue
        if show_trace:
            print(f"--- 本轮结束：{ctx.stop_reason}，共 {ctx.step_index} step ---", flush=True)
        print(f"{ctx.agent_name}：{ctx.reply or '（未回复）'}")
        if show_stats:
            print(f"  status={ctx.stop_reason}; steps={ctx.step_index}; "
                  f"estimated_tokens={ctx.num_tokens}")


def main():
    parser = argparse.ArgumentParser(description="CAgent 终端助手示例")
    parser.add_argument("--mock", action="store_true", help="使用离线脚本模型，不发起网络请求")
    parser.add_argument("--env-file", type=Path, help="可选 dotenv 文件（需要 python-dotenv）")
    parser.add_argument("--model", help="覆盖 OPENAI_MODEL 中的模型名或部署名")
    parser.add_argument("--base-url", help="OpenAI 兼容接口的基础地址，通常以 /v1 结尾")
    parser.add_argument("--session", default="demo-cli", help="会话 ID；配合 --db 可继续原会话")
    parser.add_argument("--db", type=Path, help="可选 SQLite 文件；默认仅在内存中保存历史")
    parser.add_argument("--stats", action="store_true", help="显示轮次状态和估算 token 数")
    parser.add_argument("--trace", action="store_true", help="显示每个 step 的输入、输出与工具结果")
    args = parser.parse_args()
    if args.env_file is not None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            parser.error('--env-file 需要 python-dotenv：python -m pip install "cagent[examples]"')
        load_dotenv(args.env_file, override=True, encoding="utf-8-sig")
    if args.mock:
        model = mock_model
        print("离线脚本模式：固定规则回复，不会请求模型服务。")
    else:
        base_url = (args.base_url or os.getenv("OPENAI_BASE_URL", "")).strip()
        model_name = (args.model or os.getenv("OPENAI_MODEL", "")).strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        missing = [name for name, value in (("OPENAI_BASE_URL", base_url),
                                            ("OPENAI_MODEL", model_name),
                                            ("OPENAI_API_KEY", api_key)) if not value]
        if missing:
            parser.error("请通过环境变量或 --env-file 配置：" + ", ".join(missing))
        model = make_openai_model(base_url=base_url, model_name=model_name, api_key=api_key,
                                  reasoning_effort=os.getenv("REASONING_EFFORT", "").strip() or None)
        print("真实模型模式：", model_name)
    if args.db:
        fresh = not args.db.exists()
        store = SQLiteStore(str(args.db))
        try:
            if fresh:
                store.initialize()
        except Exception:
            store.close()
            raise
    else:
        store = MemoryStore()
    try:
        asyncio.run(chat_loop(store, model, args.session,
                              show_stats=args.stats, show_trace=args.trace))
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        if isinstance(store, SQLiteStore):
            store.close()


if __name__ == "__main__":
    main()
