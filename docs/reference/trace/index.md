# orbiter.trace

OpenTelemetry-based observability for Orbiter agents and tools.

## Installation

```bash
pip install orbiter-trace
```

## Module path

```python
import orbiter.trace
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `TraceConfig` | `orbiter.trace.config` | Immutable configuration for the trace layer |
| `TraceBackend` | `orbiter.trace.config` | Supported trace export backends enum |
| `traced` | `orbiter.trace.decorator` | Decorator wrapping a function in an OpenTelemetry span |
| `span_sync` | `orbiter.trace.decorator` | Synchronous span context manager |
| `Timer` | `orbiter.trace.instrumentation` | Simple timer for measuring durations |
| `build_agent_attributes` | `orbiter.trace.instrumentation` | Build attribute dict for agent metrics |
| `build_tool_attributes` | `orbiter.trace.instrumentation` | Build attribute dict for tool metrics |
| `record_agent_run` | `orbiter.trace.instrumentation` | Record agent run metrics |
| `record_tool_step` | `orbiter.trace.instrumentation` | Record tool step metrics |
| `PromptLogger` | `orbiter.trace.prompt_logger` | Structured LLM execution logger |
| `ExecutionLogEntry` | `orbiter.trace.prompt_logger` | Structured record of a single LLM execution |
| `TokenBreakdown` | `orbiter.trace.prompt_logger` | Per-role token counts and context window analysis |
| `compute_token_breakdown` | `orbiter.trace.prompt_logger` | Compute a TokenBreakdown from message dicts |
| `estimate_tokens` | `orbiter.trace.prompt_logger` | Estimate token count from character length |
| `BaggagePropagator` | `orbiter.trace.propagation` | Extract and inject W3C Baggage headers |
| `Carrier` | `orbiter.trace.propagation` | Protocol for reading/writing propagation headers |
| `DictCarrier` | `orbiter.trace.propagation` | Carrier backed by a plain dict |
| `SpanConsumer` | `orbiter.trace.propagation` | ABC for span consumers |
| `get_baggage` | `orbiter.trace.propagation` | Return the current baggage as a dict |
| `get_baggage_value` | `orbiter.trace.propagation` | Return a single baggage value |
| `set_baggage` | `orbiter.trace.propagation` | Set a single baggage key-value pair |
| `clear_baggage` | `orbiter.trace.propagation` | Remove all baggage entries |
| `register_span_consumer` | `orbiter.trace.propagation` | Register a span consumer |
| `get_span_consumer` | `orbiter.trace.propagation` | Look up a span consumer by name |
| `list_span_consumers` | `orbiter.trace.propagation` | List all registered span consumer names |
| `dispatch_spans` | `orbiter.trace.propagation` | Send spans to all registered consumers |
| `clear_span_consumers` | `orbiter.trace.propagation` | Remove all registered span consumers |

## Submodules

- [orbiter.trace.config](config.md) -- TraceConfig, TraceBackend, semantic conventions
- [orbiter.trace.decorator](decorator.md) -- @traced decorator and span context managers
- [orbiter.trace.instrumentation](instrumentation.md) -- Metrics recording and Timer
- [orbiter.trace.prompt_logger](prompt-logger.md) -- PromptLogger with token breakdown
- [orbiter.trace.propagation](propagation.md) -- W3C Baggage propagation and span consumers
