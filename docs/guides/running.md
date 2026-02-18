# Running Agents

The `orbiter.runner` module provides the primary API for executing agents: `run()` (async), `run.sync()` (blocking), and `run.stream()` (async generator). These functions handle provider resolution, state tracking, loop detection, and multi-turn conversations.

## Basic Usage

```python
from orbiter.agent import Agent
from orbiter.runner import run

agent = Agent(name="assistant", model="openai:gpt-4o")

# Async
result = await run(agent, "What is 2 + 2?")
print(result.output)   # "2 + 2 equals 4."
print(result.steps)    # 1
print(result.usage)    # Usage(input_tokens=..., output_tokens=..., total_tokens=...)

# Sync (blocking wrapper for scripts/notebooks)
result = run.sync(agent, "What is 2 + 2?")
```

## run()

The primary async entry point. Auto-resolves providers, delegates to `call_runner` for state tracking and loop detection.

```python
async def run(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` or `Swarm` | *required* | The agent (or swarm) to execute |
| `input` | `str` | *required* | User query string |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history |
| `provider` | `Any` | `None` | LLM provider with `async complete()`. Auto-resolved from model string if `None` |
| `max_retries` | `int` | `3` | Retry attempts for transient LLM errors |
| `loop_threshold` | `int` | `3` | Consecutive identical tool-call patterns before raising a loop error |

When `provider` is `None`, Orbiter attempts to auto-resolve a provider from the agent's model string using the model registry (see [Model Providers](models.md)).

If you pass a `Swarm` instead of an `Agent`, `run()` delegates to the swarm's own `run()` method.

## run.sync()

A blocking wrapper that calls `run()` via `asyncio.run()`. Convenient for scripts, notebooks, and CLIs:

```python
result = run.sync(
    agent,
    "Summarize this document",
    messages=history,
    max_retries=5,
)
```

The parameters are identical to `run()`.

## run.stream()

An async generator that yields real-time events during agent execution:

```python
async def run.stream(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_steps: int | None = None,
) -> AsyncIterator[StreamEvent]: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Agent` | *required* | The agent to execute |
| `input` | `str` | *required* | User query string |
| `messages` | `Sequence[Message] \| None` | `None` | Prior conversation history |
| `provider` | `Any` | `None` | LLM provider with `async stream()` method |
| `max_steps` | `int \| None` | `None` | Maximum LLM-tool round-trips (defaults to `agent.max_steps`) |

**Usage:**

```python
from orbiter.runner import run
from orbiter.types import TextEvent, ToolCallEvent

async for event in run.stream(agent, "Tell me about quantum computing"):
    if isinstance(event, TextEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[Calling tool: {event.tool_name}]")
```

### Stream Events

Two event types are yielded:

**TextEvent** -- a chunk of text from the LLM:

```python
class TextEvent(BaseModel):
    type: Literal["text"] = "text"
    text: str                    # the text chunk
    agent_name: str = ""         # which agent produced it
```

**ToolCallEvent** -- notification that a tool is being called:

```python
class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str               # name of the tool
    tool_call_id: str            # correlation ID
    agent_name: str = ""         # which agent produced it
```

When tool calls are detected during streaming, tools are executed and the LLM is re-streamed with the results. This loops until a text-only response or `max_steps` is reached.

## RunResult

The return type of `run()` and `run.sync()`:

```python
class RunResult(BaseModel):
    output: str = ""                           # final text output
    messages: list[Message] = []               # full message history
    usage: Usage = Usage()                     # aggregated token usage
    steps: int = 0                             # number of LLM call steps
```

## Multi-Turn Conversations

Pass the message history from a previous result to continue the conversation:

```python
# First turn
result1 = await run(agent, "My name is Alice")

# Second turn -- pass messages from first turn
result2 = await run(agent, "What's my name?", messages=result1.messages)
print(result2.output)  # "Your name is Alice."
```

## Loop Detection

The runner detects endless loops where the agent repeatedly produces the same tool calls without making progress. When the same set of tool calls (by name and arguments) repeats `loop_threshold` consecutive times, a `CallRunnerError` is raised:

```python
from orbiter._internal.call_runner import CallRunnerError

try:
    result = await run(agent, "Do something", loop_threshold=3)
except CallRunnerError as e:
    print(f"Loop detected: {e}")
```

## Provider Auto-Resolution

When no `provider` is passed, `run()` attempts to auto-resolve one:

1. It reads `agent.provider_name` (parsed from the model string, e.g., `"openai"`)
2. It calls `get_provider(provider_name)` from the model registry
3. If resolution fails, the agent's own `run()` method raises `AgentError`

To use auto-resolution, install the `orbiter-models` package and ensure your API key is configured:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

agent = Agent(name="bot", model="openai:gpt-4o")
result = await run(agent, "Hello!")  # provider auto-resolved
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `run()` | `orbiter.runner` | Async agent execution |
| `run.sync()` | `orbiter.runner` | Blocking agent execution |
| `run.stream()` | `orbiter.runner` | Streaming agent execution |
| `RunResult` | `orbiter.types` | Return type of `run()` |
| `TextEvent` | `orbiter.types` | Streaming text chunk event |
| `ToolCallEvent` | `orbiter.types` | Streaming tool call event |
| `StreamEvent` | `orbiter.types` | Union of `TextEvent \| ToolCallEvent` |
| `Usage` | `orbiter.types` | Token usage statistics |
