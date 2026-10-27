# CAgent

Python 3.10+ 的通用 Agent 框架，只依赖标准库。核心语义：**宿主自持存储（方法可同步可异步）、图像输入（用户输入与工具结果）、工具人工审批、协作式打断、轮末压缩、可审计的消息历史**。

~~~text
src/cagent/
├── agent.py     # BaseAgent：模型 → 工具 → 模型循环、Hook 链、压缩调度
├── images.py    # 图片引用（Image）与校验，用户输入和工具结果共用
├── context.py   # 每轮新建的 Context；append_messages 先落库再加入内存
├── storage.py   # Store 协议（由宿主实现；方法可同步或异步）
├── stores.py    # MemoryStore / SQLiteStore 参考实现
├── models.py    # OpenAI 兼容 Chat Completions 适配器
└── tools/       # Tool / ToolError / ToolResult 与执行器
~~~

## 设计边界

- **宿主拥有**：数据库连接与事务、模型客户端、业务工具、部署与并发调度。
- **框架拥有**：轮次循环、工具配对与执行语义、工具审批钩子、打断检查、压缩边界、轮次记账（turn_end mark）。
- BaseAgent 只有类方法，不实例化；所有可变状态都在每轮新建的 ctx 中。
- 同一 session 的 turn 必须由调用方串行调度；RUNNING 状态不构成分布式锁。

## 快速开始

~~~python
from cagent import BaseAgent, Tool, ToolError, MemoryStore

async def finish(*, content, store, ctx, agent_cls):
    if not isinstance(content, str) or not content.strip():
        raise ToolError("content 必须是非空的最终回复")
    ctx.reply_content = content          # 完成约定：交给 ctx.reply 返回
    ctx.stop("completed")
    return f"【{ctx.agent_name}回复】{content}"

class Assistant(BaseAgent):
    NAME = "助手"
    SYSTEM_PROMPT = "你是一个简洁友好的中文助手。"

    @classmethod
    def get_tools(cls, ctx):
        return {"finish": Tool(finish, description="给出本轮最终回复并结束",
                               parameters={"type": "object",
                                           "properties": {"content": {"type": "string"}},
                                           "required": ["content"]})}

store = MemoryStore()
store.create_session("demo", "user")

async def model(*, messages, tools, tool_choice=None):
    # 适配器在这里调用真实模型；也可用 cagent.make_openai_model
    return {"tool_calls": [{"id": "c1", "name": "finish",
                            "arguments": {"content": "你好！"}}]}

# 在 async 函数中：
ctx = await Assistant.run_turn("你好", session_id="demo", store=store, model=model)
print(ctx.reply)   # 你好！
~~~

安装：`python -m pip install -e .`（示例附加依赖：`pip install -e ".[examples]"`）。

## 输入与模型回调

输入可以是非空字符串，或非空消息字典列表（整批编码为一条 user message，content 为 `{"messages": [...]}`，原始字段完整保留）。新消息进入平台 backlog 时由调用方攒批，下一轮一次性传入；本轮不读取。

模型回调接收 `messages` 和 `tools`，可同步或异步，返回框架内部格式：

~~~python
{"content": "", "tool_calls": [{"id": "unique-id", "name": "finish",
                                "arguments": {"content": "你好"}}]}
~~~

`arguments` 必须是字典（模型客户端返回 JSON 字符串时先解析）。空 content 且无工具调用表示静默结束，`ctx.reply` 为 None。`usage` 原样记录到 turn_end mark，不参与上下文统计。

### 图像输入

`run_turn(..., images=[...])` 给本轮用户输入附带图片。每项是引用字符串，或 `{"ref": ..., "detail": "auto" | "low" | "high"}`；有图时文本可以为空字符串。

~~~python
ctx = await Assistant.run_turn("这是什么？", session_id="demo", store=store, model=model,
                               images=["file:42", {"ref": "https://example.com/a.png", "detail": "low"}])
~~~

`images=` 里的图片按列表顺序排在文本之后。需要图文交替时，输入改为由字符串和 `Image` 组成的有序列表（此时不再传 `images=`）：

~~~python
from cagent import Image

ctx = await Assistant.run_turn(
    ["旧版：", Image("file:a"), " 新版：", Image("file:b", detail="low"), " 哪个更好？"],
    session_id="demo", store=store, model=model)
~~~

列表里只要出现字符串或 `Image`，就按交替输入处理；全是字典时仍是原来的消息批量输入。

- 入库只存引用（`metadata.images`），不存图片字节；历史中的图片在后续轮次仍会发给模型。交替输入的文本片段拼接后存为 `content`，每张图额外记录 `at`（在 `content` 中的字符位置），只读文本的消费者不受影响。
- 每次调用模型前，`resolve_image(store, ctx, image)` 把引用解析成模型可读的 URL，结果在本轮内缓存。默认只放行 `http(s)://` 和 `data:image/` 引用；宿主用文件 ID 时覆盖它，返回 data URL 或签名 URL：

~~~python
class Assistant(BaseAgent):
    @classmethod
    async def resolve_image(cls, store, ctx, image):
        data = await my_blob_store.read(image["ref"])      # 宿主自己的文件存储
        return "data:image/png;base64," + base64.b64encode(data).decode()
~~~

- 发给模型的是 OpenAI 风格的 content parts（`text` + `image_url`），`make_openai_model` 原样透传。
- 压缩时图片随历史一起交给压缩模型，由摘要文本描述；压缩生效后，边界前的图片不再发送。
- 工具结果也可以带图，见下文“工具返回图片”。默认 token 估算只计引用长度、不计图片本身；`COUNT_TOKENS` 收到的是未解析的 `image_ref` 部分，需要准确统计时在其中自行计入图片开销。


## 工具

工具通过 `run_turn(tools=...)` 传入，或由子类 `get_tools(ctx)` 提供。工具函数接收模型参数和框架注入的 `store`、`ctx`、`agent_cls`（三者不可被模型参数覆盖）；同步异步均可，按模型给出的顺序串行执行。

- 未知工具、参数不匹配和 `ToolError` 以纯文本错误结果交给下一步模型；意外异常向外抛出（可能有副作用，不自动重试）。
- 字符串返回值原样写入 tool 消息；其他返回值须可 JSON 序列化。
- 需要给消息附加宿主数据时返回 `ToolResult(content="正文", metadata={"ids": [1]})`，metadata 只入库不发给模型。

### 工具返回图片

工具返回 `ToolResult(content="已截图", images=["file:42", {"ref": "https://...", "detail": "low"}])` 即可把图片交给模型。`images` 的写法与用户输入图片相同（引用字符串、`Image` 或 `{ref, detail}`），在构造时校验；入库时存为 tool 消息的 `metadata.images`（只存引用，`"images"` 因此是保留键，不能再放进 `metadata`）。

- 发给模型时，tool 消息的 content 变为 `[文本, image_url, ...]`，引用同样经 `resolve_image` 解析、本轮内缓存。出错、被拒绝或被跳过的工具结果不带图。
- 图文交错：`content` 也可以和用户输入一样传交错列表，`ToolResult(["第 1 页：", Image("file:1"), "第 2 页：", Image("file:2", detail="low")], metadata={...})`，模型看到的顺序与列表一致。构造后 `content` 是拼接后的文本，`images` 中每张图带 `at`（在文本中的字符偏移）；交错写法不能再同时传 `images=`。被 `TOOL_IMAGE_LIMIT` 省略的图片连同位置一起去掉。`tool_images="user_message"` 模式下图片会集中移到后续 user 消息，交错顺序不保留。
- `TOOL_IMAGE_LIMIT = N` 只把最近 N 张工具图片发给模型（`None` 全部保留，`0` 全部不发），更早的替换为“[k earlier tool image(s) omitted]”文字；用户图片不受影响，入库的引用也不删除。压缩时同样遵守该限制。浏览器截图这类每步都产生图片的工具建议设一个小值。
- 线上格式：Chat Completions 已发布的 schema 只允许 tool 消息带文本，Responses API 的 `function_call_output` 则支持图片；很多网关在 Chat Completions 下也接受 tool 消息带图。`make_openai_model(tool_images="inline")`（默认）保持图片在 tool 消息内；严格按 schema 校验的端点用 `tool_images="user_message"`，图片会移到这批工具结果之后的一条 user 消息里，并注明是工具输出而非用户指令。自己写模型适配器的宿主可直接调用 `relocate_tool_images(messages)` 得到同样效果。
- 工具用 `Tool.result_type` 声明成功结果的消息分类（signal / memory / extra，默认 extra）；失败结果一律 extra。分类跟随工具定义，换注册名仍有效。
- 工具调用 ID 在当前历史中必须唯一，复用会被拒绝。

### 完成约定

框架不内置任何业务工具，但保留统一的完成语义：工具设置 `ctx.reply_content` 并 `ctx.stop("completed")`；`ctx.reply` 优先返回它，否则取最后一条 assistant 非空文本。`examples/cli.py` 的 `finish` 工具是最小实现。

### 停止后的配对保护

某工具停止本轮（如 finish）或平台请求打断后，同批剩余调用不执行，但仍写入“本轮已停止，未执行工具 X”的配对结果，保证下一步模型输入的工具历史完整有效。

### 工具审批（Human-in-the-loop）

覆盖 `approve_tool_call(store, ctx, tool_call, tool)` 即可为每个工具调用加人工确认。返回 `None` 或 `True` 放行；返回其他值拒绝执行——非空字符串会作为拒绝原因交给模型，其余值使用通用文案。拒绝与跳过一样产出配对的 tool 结果，工具历史始终有效；未知工具不经过审批，直接按错误返回。

钩子可以 await 外部决策（人工确认可能耗时任意久）；返回后框架会重新检查打断状态，等待期间发生的打断不会漏掉。等待期间的会话状态由宿主掌握：可设置等待状态（如 WAITING_CONFIRM），决策结束后调用 `restore_running(store, ctx)` 恢复 RUNNING——它会先确认平台没有在等待期间置为 INTERRUPTED，避免覆盖打断。想在决策后给模型补充指令，向 `ctx.pending_messages` 追加文本，它会在这批工具结果之后、下一步模型调用之前写入。

~~~python
class GuardedAgent(BaseAgent):
    @classmethod
    async def approve_tool_call(cls, store, ctx, tool_call, tool):
        if tool_call["name"] != "bash":
            return None                                # 低风险工具直接放行
        store.set_status(ctx.session_id, "WAITING_CONFIRM")
        try:
            approved, note = await ask_human(tool_call)  # 宿主自己的确认通道
        finally:
            await cls.restore_running(store, ctx)
        if note:
            ctx.pending_messages.append(note)           # 补充指令，稍后交给模型
        return None if approved else "用户拒绝执行该命令"
~~~

## Hook 与生命周期

~~~text
pre_run_turn(store, session_id, input_message, max_steps) → 新 ctx
    should_continue(store, ctx)          # async
    pre_run_step(store, ctx)
    should_continue(store, ctx)
    run_step(store, ctx, model)
      approve_tool_call(store, ctx, tool_call, tool)   # 每个工具执行前，async
    post_run_step(store, ctx)
    ...循环...
should_compress_turn(ctx) → 满足条件则 compress_turn(...)   # 本轮压缩
record_turn(store, ctx, count_tokens)
should_compress(ctx) → 满足条件则 launch_compression_task(...)   # 全局压缩
post_run_turn(store, ctx)            # 移除临时标记并释放状态
~~~

所有 Hook 均为 classmethod；除 `should_compress` 为同步外，其余均为 async。覆盖时按需调用 `super()`。`on_event` 回调接收 step_start / step_end 事件副本，异常只记日志不影响对话。

`ctx.stop_reason` 可能为 completed、step_limit、interrupted、cancelled、failed；仅 completed 时 `ctx.reply` 有值。开始执行时会话状态 RUNNING；正常结束或步数上限后 IDLE；取消与打断为 INTERRUPTED；失败为 FAILED。

### 协作式打断

平台随时 `store.set_status(session_id, "INTERRUPTED")`。当前模型/工具等待结束后检查状态：不再执行后续工具，未执行的工具补配对结果，`ctx.reply` 为 None。取消任务向外抛出 CancelledError。若失败或取消留下未配对历史，下次 turn 会明确拒绝，需调用方核实并修复，框架不会自动重放可能已执行的动作。

### pipeline

`pipeline` 是每步一条的提示指令列表（以 type=extra 的 user 消息写入），不是强制工具调用机制；`[]` 表示自由模型/工具循环；不传则使用类属性 `PIPELINE`。指令用完后仍可继续读取工具结果并生成最终回复；有待执行指令时，普通文本输出不会提前结束流程。工具可用 `ctx.stop()` 显式结束。

## 轮末压缩

每轮结束写入 turn_end mark（stop_reason、num_tokens、usage 等）。三个条件任一满足即触发：有效上下文 token 达到 `COMPRESSION_TOKEN_THRESHOLD`、上次压缩后完成轮数达到 `COMPRESSION_TURN_THRESHOLD`、`FORCE_COMPRESSION=True`。阈值设 None 关闭；默认值仅为占位，应按实际模型调小。

压缩流程：以 compression_started mark 固定截止某个 turn_end 的边界 → 总结边界前历史（禁止工具调用）→ 校验摘要非空、无工具调用、确实缩小上下文 → 原子写入 compression_completed mark 才生效，原始记录不删除。摘要作为带前缀的 user message 呈现，不会成为 system 指令。压缩失败（网络、空摘要、过长、写入失败、打断）时旧上下文保持有效，`ctx.compression_error` 说明原因，已完成的回复仍可返回。

压缩请求的格式：与普通模型调用相同的前缀（Agent 的 system prompt + 投影后的历史），末尾追加一条内容为 `COMPRESSION_INSTRUCTION`（单轮压缩为 `TURN_COMPRESSION_INSTRUCTION`）的 user 消息，工具见下方 `COMPRESSION_SEND_TOOLS`。前缀一致便于复用提示缓存；请求总以 user 消息结尾，兼容不支持 assistant 预填充（prefill）的模型（如 Bedrock 上的 Claude）。压缩指令不再作为 system message 发送。

`COMPRESSION_SEND_TOOLS`（默认 True）：两种压缩请求都带上 Agent 的真实工具定义并传 `tool_choice="none"`，在接口层面禁止调用工具。只传 `tools=[]` 并不能阻止工具调用：历史里满是工具调用时，不少端点仍会让模型返回工具调用（部分网关会为这类历史自动补工具配置），压缩随之失败。带上同样的工具定义还让请求前缀与普通步骤完全一致，更易命中提示缓存。模型回调须接受 `tool_choice` 关键字参数（`make_openai_model` 已支持，且只在带工具时发送）；设为 False 恢复旧行为（`tools=[]`，不传 `tool_choice`），适合只接受两个参数的旧回调。

### 本轮压缩

`TURN_COMPRESSION_TOOL_CALL_THRESHOLD`（默认 None 关闭）：单轮工具调用次数达到该值时，在写 turn_end 之前把本轮压缩为一条摘要。`pre_run_turn` 在内存中的 `ctx.messages` 里插入一个临时标记 `tmp_turn_start_mark`（从不写入 Store）；轮末若 stop_reason 为 completed 或 step_limit 且满足阈值，`compress_turn` 把标记替换为 `TURN_COMPRESSION_MARKER` 提示语，连同之前的上下文与标记之后的全部消息（含本轮用户输入与最终回复）以 `TURN_COMPRESSION_INSTRUCTION` 发给压缩模型。摘要以 name=turn_summary 的 user 消息写入，其 `metadata.replaced_message_ids` 列出被替换的消息，这次写入即原子生效；之后被替换消息被软删除。摘要无效、未缩小上下文或被打断时保留原始本轮记录，`ctx.turn_compression_error` 说明原因。无论是否压缩，`post_run_turn` 都会移除临时标记。本轮压缩完成后才统计 token 并判断全局压缩，因此全局阈值看到的是压缩后的上下文。

`COUNT_TOKENS` 可注入与实际模型匹配的 tokenizer（计入适配器额外注入的内容）；默认按 UTF-8 JSON 字节数除以 3 粗略估算。空闲期可绕过用户输入直接压缩：

~~~python
ctx = await Assistant.compress_session(
    session_id=session_id, store=store, model=summary_model,
    system_prompt="你是一个简洁友好的中文助手。")  # 与正常对话一致，用于统计
~~~

## 模型适配器

`make_openai_model(base_url=..., model_name=..., api_key=..., reasoning_effort=None, tool_images="inline")` 返回兼容任何 OpenAI 风格端点（vLLM、Ollama、OpenRouter、各类代理网关等）的异步模型 `model(*, messages, tools, tool_choice=None)`，负责工具定义、调用与结果的线上格式转换；请求在后台线程执行。`reasoning_effort` 仅在提供时发送；`tool_images` 见“工具返回图片”；`ModelError` 不回显响应体。人设通过 `system_prompt` 参数传入，适配器不额外添加。

## Store 与消息

store 是调用方传入的数据访问对象，框架不创建、初始化或关闭它；实现无须继承 `Store` 协议，满足方法约定即可。协议方法可同步可异步、可以混用：异步实现（例如包一层 async 数据库驱动）的返回值由框架透明 await；宿主在自己的 Hook 里需要同样行为时可用 `maybe_await`。消息为普通字典，`type` 是与模型 `role` 独立的业务分类（signal / memory / extra / None），Store 保留显式值、不按 role 猜测。框架保留 mark 名称 turn_end、compression_started、compression_completed，以及 user 消息名称 turn_summary。

`stores.py` 提供 MemoryStore（测试/演示）与 SQLiteStore（单连接、短事务的文件参考实现，`initialize()` 显式建表）；生产服务应针对自己的数据库实现同一协议，连接与事务生命周期放在实现内部。`ctx.messages` 保留全部历史（含软删除）；`ctx.model_messages()` 按已提交摘要裁剪、过滤 mark 与软删除、校验工具配对后才交给模型。

## 示例与验证

~~~sh
python examples/cli.py --mock --trace     # 离线：工具循环 + 完成约定 + 历史
python examples/cli.py                    # 读取 OPENAI_BASE_URL/MODEL/API_KEY 调真实模型
python examples/cli.py --mock --db chat.db --session demo   # SQLite 持久化，重开续聊
python -m unittest discover -s tests -v
~~~

示例支持 /history、/new、/exit、--stats、--trace；也可用 `--env-file`（需 python-dotenv）加载配置。

## 路线

当前是单 agent、单 session 内循环；多 agent 编排、MCP 工具桥、流式输出均未包含，需要时在宿主层组合或后续迭代。
