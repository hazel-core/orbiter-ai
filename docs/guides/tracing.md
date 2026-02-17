# Tracing

The `orbiter-trace` package provides OpenTelemetry-based observability for agent execution. It includes a `@traced` decorator for automatic span creation, semantic conventions for agent/tool/LLM metrics, prompt logging with token breakdowns, W3C baggage propagation, and a plugin system for span consumers.

## Basic Usage

```python
from orbiter.trace import traced

@traced(name="research_task")
async def research(topic: str) -> str:
    """A traced function -- automatically creates a span."""
    # ... do work ...
    return f"Results for {topic}"

# When called, this creates an OpenTelemetry span named "research_task"
result = await research("quantum computing")
```

## The @traced Decorator

The `@traced` decorator wraps sync functions, async functions, and async generators with automatic span creation:

```python
from orbiter.trace import traced

# Async function
@traced(name="fetch_data", attributes={"source": "api"})
async def fetch_data(url: str) -> str:
    return await http_get(url)

# Sync function
@traced(name="process")
def process_data(data: str) -> str:
    return data.upper()

# Async generator (streaming)
@traced(name="stream_results")
async def stream_results(query: str):
    for chunk in chunks:
        yield chunk
```

### Decorator Parameters

```python
@traced(
    name="my_operation",          # span name (defaults to function name)
    attributes={"key": "value"},  # static span attributes
    extract_args=True,            # include function args as attributes
)
```

When `extract_args=True`, the decorator uses `extract_metadata()` to pull function arguments into span attributes.

### Span Context Managers

For manual span management without decorating a function:

```python
from orbiter.trace.decorator import span_sync, span_async

# Sync context manager
with span_sync("my_sync_operation", attributes={"step": 1}):
    do_sync_work()

# Async context manager
async with span_async("my_async_operation"):
    await do_async_work()
```

## Configuration

```python
from orbiter.trace import TraceConfig, TraceBackend

config = TraceConfig(
    service_name="my-agent-app",
    backend=TraceBackend.OTLP,
    endpoint="http://localhost:4317",
    sample_rate=1.0,
    enabled=True,
)
```

### Backends

| Backend | Value | Description |
|---------|-------|-------------|
| `OTLP` | `"otlp"` | Export spans to an OTLP collector (Jaeger, Grafana, etc.) |
| `MEMORY` | `"memory"` | Store spans in-memory (testing) |
| `CONSOLE` | `"console"` | Print spans to stdout (debugging) |

## Semantic Conventions

The package defines OpenTelemetry semantic conventions for agent framework spans:

### GenAI Attributes

| Attribute | Description |
|-----------|-------------|
| `gen_ai.system` | LLM provider (e.g., "openai") |
| `gen_ai.request.model` | Model name |
| `gen_ai.request.temperature` | Temperature setting |
| `gen_ai.request.max_tokens` | Max tokens setting |
| `gen_ai.response.model` | Model used in response |
| `gen_ai.usage.prompt_tokens` | Prompt token count |
| `gen_ai.usage.completion_tokens` | Output token count |

### Agent Attributes

| Attribute | Description |
|-----------|-------------|
| `orbiter.agent.name` | Agent name |
| `orbiter.agent.step` | Current step number |
| `orbiter.agent.status` | Completion status |

### Tool Attributes

| Attribute | Description |
|-----------|-------------|
| `orbiter.tool.name` | Tool name |
| `orbiter.tool.status` | Execution status |
| `orbiter.tool.error` | Error message (if any) |

## Instrumentation Helpers

Record structured spans for agent runs and tool executions:

```python
from orbiter.trace.instrumentation import record_agent_run, record_tool_step, Timer

# Time an operation
timer = Timer()
timer.start()
# ... do work ...
elapsed = timer.stop()

# Record an agent run span
record_agent_run(
    agent_name="researcher",
    step=1,
    model="gpt-4o",
    prompt_tokens=500,
    output_tokens=150,
    elapsed_ms=elapsed * 1000,
)

# Record a tool execution span
record_tool_step(
    tool_name="search",
    status="success",
    elapsed_ms=120.5,
)
```

### Building Attributes

```python
from orbiter.trace.instrumentation import build_agent_attributes, build_tool_attributes

agent_attrs = build_agent_attributes(
    name="researcher",
    step=1,
    model="gpt-4o",
)
# {"orbiter.agent.name": "researcher", "orbiter.agent.step": 1, ...}

tool_attrs = build_tool_attributes(
    name="search",
    status="success",
)
# {"orbiter.tool.name": "search", "orbiter.tool.status": "success"}
```

## Prompt Logger

Log prompts with token breakdowns for debugging and analysis:

```python
from orbiter.trace import PromptLogger, compute_token_breakdown

# Compute per-role token breakdown
breakdown = compute_token_breakdown(messages)
print(f"System: {breakdown.system_tokens}")
print(f"User: {breakdown.user_tokens}")
print(f"Assistant: {breakdown.assistant_tokens}")
print(f"Tool: {breakdown.tool_tokens}")

# Logger tracks execution entries
logger = PromptLogger()
logger.log(
    messages=messages,
    response=response_text,
    model="gpt-4o",
    tokens=breakdown,
)
```

`TokenBreakdown` provides per-role token counts:

```python
@dataclass(frozen=True)
class TokenBreakdown:
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    tool_tokens: int
    total_tokens: int
```

## Propagation

### W3C Baggage

Propagate context across service boundaries using W3C Baggage:

```python
from orbiter.trace.propagation import BaggagePropagator, DictCarrier

propagator = BaggagePropagator()

# Inject baggage into outgoing request headers
carrier = DictCarrier()
propagator.inject(carrier)

# Extract baggage from incoming request headers
propagator.extract(incoming_headers)
```

### Baggage Context Variables

```python
from orbiter.trace.propagation import set_baggage, get_baggage, clear_baggage

# Set baggage values
set_baggage("user_id", "u-123")
set_baggage("session_id", "s-456")

# Read baggage
user_id = get_baggage("user_id")  # "u-123"

# Clear all baggage
clear_baggage()
```

## Span Consumer Plugin System

Register custom consumers that process spans after creation:

```python
from orbiter.trace.propagation import SpanConsumer, register_consumer

class LogConsumer(SpanConsumer):
    """Log all spans to a file."""

    def consume(self, span_data: dict) -> None:
        with open("spans.log", "a") as f:
            f.write(f"{span_data}\n")

# Register the consumer
register_consumer(LogConsumer())
```

## Advanced Patterns

### End-to-End Tracing

Trace an entire agent pipeline from input to output:

```python
@traced(name="agent_pipeline", extract_args=True)
async def run_pipeline(input: str) -> str:
    # Each sub-step creates a child span
    context = await prepare_context(input)
    response = await call_llm(context)
    result = await execute_tools(response)
    return result

@traced(name="prepare_context")
async def prepare_context(input: str) -> dict:
    ...

@traced(name="call_llm")
async def call_llm(context: dict) -> str:
    ...

@traced(name="execute_tools")
async def execute_tools(response: str) -> str:
    ...
```

### Custom Metric Collection

Use the instrumentation module to record custom metrics:

```python
from orbiter.trace.instrumentation import Timer

async def timed_operation():
    timer = Timer()
    timer.start()

    result = await expensive_operation()

    elapsed = timer.stop()
    record_agent_run(
        agent_name="worker",
        step=1,
        elapsed_ms=elapsed * 1000,
        prompt_tokens=0,
        output_tokens=0,
    )
    return result
```

### Distributed Tracing with A2A

When using Agent-to-Agent communication, propagate trace context:

```python
from orbiter.trace.propagation import BaggagePropagator, DictCarrier

# Sender: inject context into request
carrier = DictCarrier()
propagator = BaggagePropagator()
propagator.inject(carrier)
# Send carrier.items as HTTP headers

# Receiver: extract context from request
propagator.extract(request.headers)
# Spans created here are linked to the sender's trace
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `traced` | `orbiter.trace` | Decorator for automatic span creation |
| `TraceConfig` | `orbiter.trace` | Configuration: backend, endpoint, sample rate |
| `TraceBackend` | `orbiter.trace` | Enum: `OTLP`, `MEMORY`, `CONSOLE` |
| `PromptLogger` | `orbiter.trace` | Log prompts with token breakdowns |
| `TokenBreakdown` | `orbiter.trace.prompt_logger` | Per-role token counts |
| `compute_token_breakdown` | `orbiter.trace.prompt_logger` | Compute breakdown from messages |
| `record_agent_run` | `orbiter.trace.instrumentation` | Record agent execution span |
| `record_tool_step` | `orbiter.trace.instrumentation` | Record tool execution span |
| `Timer` | `orbiter.trace.instrumentation` | Simple start/stop timer |
| `BaggagePropagator` | `orbiter.trace.propagation` | W3C Baggage context propagation |
| `SpanConsumer` | `orbiter.trace.propagation` | ABC for span processing plugins |
| `register_consumer` | `orbiter.trace.propagation` | Register a span consumer |
| `span_sync` | `orbiter.trace.decorator` | Sync context manager for manual spans |
| `span_async` | `orbiter.trace.decorator` | Async context manager for manual spans |
