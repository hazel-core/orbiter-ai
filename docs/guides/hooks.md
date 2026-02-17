# Hooks

Hooks intercept the agent lifecycle at specific points, allowing you to add logging, metrics, validation, or custom behavior without modifying agent code. Hooks are async functions that run sequentially in registration order. Unlike [events](events.md), hook exceptions are **not** suppressed -- a failing hook aborts the agent run.

## Basic Usage

```python
from orbiter.agent import Agent
from orbiter.hooks import HookPoint

async def log_llm_call(**data):
    agent = data.get("agent")
    messages = data.get("messages", [])
    print(f"[{agent.name}] LLM call with {len(messages)} messages")

async def log_llm_response(**data):
    agent = data.get("agent")
    response = data.get("response")
    print(f"[{agent.name}] LLM responded: {response.content[:50]}...")

agent = Agent(
    name="hooked_agent",
    hooks=[
        (HookPoint.PRE_LLM_CALL, log_llm_call),
        (HookPoint.POST_LLM_CALL, log_llm_response),
    ],
)
```

## HookPoint Enum

The `HookPoint` enum defines the seven lifecycle points where hooks can be attached:

```python
class HookPoint(enum.Enum):
    START = "start"
    FINISHED = "finished"
    ERROR = "error"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
```

| HookPoint | When it fires | Data passed |
|-----------|---------------|-------------|
| `START` | Run begins | `agent` |
| `FINISHED` | Run completes successfully | `agent`, result data |
| `ERROR` | Run fails with an exception | `agent`, error info |
| `PRE_LLM_CALL` | Before each LLM API call | `agent`, `messages` |
| `POST_LLM_CALL` | After each LLM API call | `agent`, `response` |
| `PRE_TOOL_CALL` | Before each tool execution | `agent`, `tool_name`, `arguments` |
| `POST_TOOL_CALL` | After each tool execution | `agent`, `tool_name`, `result` |

## Hook Type

A hook is any async callable that accepts keyword arguments:

```python
from orbiter.hooks import Hook
# Hook = Callable[..., Coroutine[Any, Any, None]]

async def my_hook(**data):
    # data contains the keyword arguments for this hook point
    pass
```

## HookManager API

The `HookManager` class manages hook registration and execution. Each `Agent` has its own `HookManager` instance at `agent.hook_manager`.

### Creating a HookManager

```python
from orbiter.hooks import HookManager, HookPoint

manager = HookManager()
```

### Registering Hooks

```python
async def on_pre_llm(**data):
    print("About to call LLM")

# Via HookManager directly
manager.add(HookPoint.PRE_LLM_CALL, on_pre_llm)

# Via Agent constructor
agent = Agent(
    name="bot",
    hooks=[(HookPoint.PRE_LLM_CALL, on_pre_llm)],
)

# After construction
agent.hook_manager.add(HookPoint.POST_LLM_CALL, on_post_llm)
```

### Removing Hooks

```python
manager.remove(HookPoint.PRE_LLM_CALL, on_pre_llm)
# Silently does nothing if the hook is not registered
# Removes only the first occurrence
```

### Running Hooks

```python
await manager.run(HookPoint.PRE_LLM_CALL, agent=agent, messages=msg_list)
# Hooks run sequentially in registration order
# Exceptions propagate immediately
```

### Checking for Hooks

```python
if manager.has_hooks(HookPoint.PRE_LLM_CALL):
    print("There are pre-LLM hooks registered")
```

### Clearing All Hooks

```python
manager.clear()
# Removes all hooks for all lifecycle points
```

### Convenience Function

```python
from orbiter.hooks import run_hooks

await run_hooks(manager, HookPoint.PRE_LLM_CALL, agent=agent, messages=msg_list)
```

## Hook Lifecycle During Agent Execution

During a single agent `run()`, hooks fire in this order:

```
1. PRE_LLM_CALL  (before LLM API call)
2. POST_LLM_CALL (after LLM API call)
   If tool calls present:
3.   PRE_TOOL_CALL  (for each tool)
4.   POST_TOOL_CALL (for each tool)
5.   PRE_LLM_CALL   (before next LLM call)
6.   POST_LLM_CALL  (after next LLM call)
   ... repeat until no tool calls or max_steps reached
```

## Practical Examples

### Logging Hook

```python
import json

async def log_everything(**data):
    agent = data.get("agent")
    tool_name = data.get("tool_name")
    result = data.get("result")
    if tool_name and result:
        print(f"[{agent.name}] Tool '{tool_name}' returned: {result.content[:100]}")

agent = Agent(
    name="logged_agent",
    hooks=[
        (HookPoint.POST_TOOL_CALL, log_everything),
    ],
)
```

### Token Budget Hook

```python
class TokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0

    async def check_budget(self, **data):
        response = data.get("response")
        if response and response.usage:
            self.used += response.usage.total_tokens
            if self.used > self.max_tokens:
                raise RuntimeError(f"Token budget exceeded: {self.used}/{self.max_tokens}")

budget = TokenBudget(max_tokens=10000)

agent = Agent(
    name="budgeted",
    hooks=[
        (HookPoint.POST_LLM_CALL, budget.check_budget),
    ],
)
```

### Tool Approval Hook

```python
DANGEROUS_TOOLS = {"delete_file", "drop_table"}

async def approve_dangerous_tools(**data):
    tool_name = data.get("tool_name", "")
    if tool_name in DANGEROUS_TOOLS:
        # In a real app, this would prompt a human
        raise RuntimeError(f"Blocked dangerous tool: {tool_name}")

agent = Agent(
    name="safe_agent",
    hooks=[
        (HookPoint.PRE_TOOL_CALL, approve_dangerous_tools),
    ],
)
```

## Hooks vs Events

| Feature | Hooks | Events |
|---------|-------|--------|
| **Purpose** | Lifecycle interception | Decoupled communication |
| **Error handling** | Exceptions propagate (abort run) | Exceptions are swallowed |
| **Registration** | Per-agent via `HookManager` | Global via `EventBus` |
| **Execution** | Sequential, in order | Sequential, in order |
| **Use case** | Validation, budget enforcement | Logging, metrics, notifications |

See the [Events guide](events.md) for details on the event bus.

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `HookPoint` | `orbiter.hooks` | Enum of lifecycle interception points |
| `Hook` | `orbiter.hooks` | Type alias for async hook functions |
| `HookManager` | `orbiter.hooks` | Manages hook registration and execution |
| `run_hooks()` | `orbiter.hooks` | Convenience function to run hooks |
