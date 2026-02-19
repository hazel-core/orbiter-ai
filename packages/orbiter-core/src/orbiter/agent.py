"""Agent class: the core autonomous unit in Orbiter."""

from __future__ import annotations

import asyncio
import importlib
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from orbiter._internal.message_builder import build_messages
from orbiter._internal.output_parser import parse_response, parse_tool_arguments
from orbiter.config import parse_model_string
from orbiter.hooks import Hook, HookManager, HookPoint
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from orbiter.tool import Tool, ToolError
from orbiter.types import (
    AgentOutput,
    AssistantMessage,
    Message,
    OrbiterError,
    SystemMessage,
    ToolResult,
    UserMessage,
)

_log = get_logger(__name__)

# Sentinels: distinguish "not provided" (auto-create) from explicit None (disable)
_MEMORY_UNSET: Any = object()
_CONTEXT_UNSET: Any = object()


def _make_default_memory() -> Any:
    """Try to create a default AgentMemory. Returns None if orbiter-memory is not installed."""
    try:
        from orbiter.memory.backends.sqlite import SQLiteMemoryStore  # pyright: ignore[reportMissingImports]
        from orbiter.memory.base import AgentMemory  # pyright: ignore[reportMissingImports]
        from orbiter.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

        return AgentMemory(short_term=ShortTermMemory(), long_term=SQLiteMemoryStore())
    except ImportError:
        return None


def _make_default_context() -> Any:
    """Try to create a default ContextConfig(mode='copilot'). Returns None if not installed."""
    try:
        from orbiter.context.config import make_config  # pyright: ignore[reportMissingImports]

        return make_config("copilot")
    except ImportError:
        return None


def _make_context_from_mode(mode: Any) -> Any:
    """Create a ContextConfig from a mode string or AutomationMode enum.

    Returns None if orbiter-context is not installed.
    """
    try:
        from orbiter.context.config import make_config  # pyright: ignore[reportMissingImports]

        return make_config(mode)
    except ImportError:
        return None


class _ProviderSummarizer:
    """Wraps a model provider for use with orbiter-memory's generate_summary()."""

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def summarize(self, prompt: str) -> str:
        """Call provider.complete() to generate a summary string."""
        try:
            response = await self._provider.complete(
                [UserMessage(content=prompt)],
                tools=None,
                temperature=0.3,
                max_tokens=512,
            )
            return str(response.content or "")
        except Exception as exc:
            _log.warning("Context summarization provider call failed: %s", exc)
            return ""


async def _apply_context_windowing(
    msg_list: list[Message],
    context: Any,
    provider: Any,
) -> list[Message]:
    """Apply ContextConfig windowing and optional summarization to msg_list.

    Called before the LLM call when context is set. Applies (in order):

    1. Offload threshold: when exceeded, aggressively trims to summary_threshold
    2. Summary threshold: when exceeded, attempts LLM summarization via orbiter-memory
    3. History windowing: always applied last, keeps last history_rounds messages

    Falls back gracefully when orbiter-memory is not installed.
    """
    history_rounds: int = getattr(context, "history_rounds", 20)
    summary_threshold: int = getattr(context, "summary_threshold", 10)
    offload_threshold: int = getattr(context, "offload_threshold", 50)

    # Separate system messages from conversation history
    system_msgs: list[Message] = [m for m in msg_list if isinstance(m, SystemMessage)]
    non_system: list[Message] = [m for m in msg_list if not isinstance(m, SystemMessage)]
    msg_count = len(non_system)

    # 1. Offload threshold: aggressive trim when far over limit
    if msg_count > offload_threshold:
        _log.debug(
            "context offload: %d messages > offload_threshold=%d, trimming to %d",
            msg_count, offload_threshold, summary_threshold,
        )
        non_system = non_system[-summary_threshold:]
        msg_count = len(non_system)

    # 2. Summary threshold: attempt summarization via orbiter-memory
    elif msg_count >= summary_threshold:
        try:
            from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
                AIMemory,
                HumanMemory,
                MemoryItem,
                ToolMemory,
            )
            from orbiter.memory.summary import (  # pyright: ignore[reportMissingImports]
                SummaryConfig,
                check_trigger,
                generate_summary,
            )

            # Convert messages to MemoryItems for trigger check
            items: list[MemoryItem] = []
            for msg in non_system:
                content = str(getattr(msg, "content", "") or "")
                if isinstance(msg, UserMessage):
                    items.append(HumanMemory(content=content))
                elif isinstance(msg, AssistantMessage):
                    items.append(AIMemory(content=content))
                else:
                    items.append(ToolMemory(content=content))

            keep_recent = max(2, summary_threshold // 2)
            summary_cfg = SummaryConfig(
                message_threshold=summary_threshold,
                keep_recent=keep_recent,
            )
            trigger = check_trigger(items, summary_cfg)

            if trigger.triggered and provider is not None:
                summarizer = _ProviderSummarizer(provider)
                result = await generate_summary(items, summary_cfg, summarizer)
                if result.summaries:
                    summary_text = "\n\n".join(result.summaries.values())
                    keep_count = len(result.compressed_items)
                    recent_msgs = non_system[-keep_count:] if keep_count > 0 else []
                    summary_msg = SystemMessage(
                        content=f"[Conversation Summary]\n{summary_text}"
                    )
                    non_system = [summary_msg] + recent_msgs
                    msg_count = len(non_system)
                    _log.debug(
                        "context summarization applied: %d -> %d messages (summary + %d recent)",
                        len(items), msg_count, keep_count,
                    )
        except ImportError:
            pass

    # 3. History windowing: keep last history_rounds messages
    if msg_count > history_rounds:
        _log.debug(
            "context windowing: trimming %d -> %d messages (history_rounds=%d)",
            msg_count, history_rounds, history_rounds,
        )
        non_system = non_system[-history_rounds:]

    return system_msgs + non_system


class AgentError(OrbiterError):
    """Raised for agent-level errors (duplicate tools, invalid config, etc.)."""


class Agent:
    """An autonomous LLM-powered agent with tools and lifecycle hooks.

    Agents are the core building block in Orbiter. Each agent wraps an LLM
    model, a set of tools, optional handoff targets, and lifecycle hooks.
    The ``run()`` method (added in a later session) executes the agent's
    tool loop.

    All parameters are keyword-only; only ``name`` is required.

    Args:
        name: Unique identifier for this agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt. Can be a string or an async callable
            that receives a context dict and returns a string.
        tools: Tools available to this agent.
        handoffs: Other agents this agent can delegate to via handoff.
        hooks: Lifecycle hooks as ``(HookPoint, Hook)`` tuples.
        output_type: Pydantic model class for structured output validation.
        max_steps: Maximum LLM-tool round-trips before stopping.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
        memory: Optional memory store for persistent memory across sessions.
        context: Optional context engine for hierarchical state and prompt building.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str = "openai:gpt-4o",
        instructions: str | Callable[..., Any] = "",
        tools: list[Tool] | None = None,
        handoffs: list[Agent] | None = None,
        hooks: list[tuple[HookPoint, Hook]] | None = None,
        output_type: type[BaseModel] | None = None,
        max_steps: int = 10,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        memory: Any = _MEMORY_UNSET,
        context_mode: Any = _CONTEXT_UNSET,
        context: Any = _CONTEXT_UNSET,
    ) -> None:
        if max_steps < 1:
            raise AgentError(f"max_steps must be >= 1, got {max_steps}")
        self.name = name
        self.model = model
        self.provider_name, self.model_name = parse_model_string(model)
        self.instructions = instructions
        self.output_type = output_type
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Auto-create AgentMemory when not explicitly specified; None disables memory
        if memory is _MEMORY_UNSET:
            memory = _make_default_memory()
            self._memory_is_auto: bool = True
        else:
            self._memory_is_auto = False
        self.memory: Any = memory
        self.conversation_id: str | None = None
        # Resolve context: explicit context takes precedence over context_mode; both
        # unset triggers auto-creation of ContextConfig(mode='copilot').
        if context is not _CONTEXT_UNSET:
            self.context = context
            self._context_is_auto: bool = False
        elif context_mode is not _CONTEXT_UNSET:
            self.context = None if context_mode is None else _make_context_from_mode(context_mode)
            self._context_is_auto = False
        else:
            self.context = _make_default_context()
            self._context_is_auto = True
        self._memory_persistence: Any = None

        # Tools indexed by name for O(1) lookup during execution
        self.tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self._register_tool(t)

        # Handoff targets indexed by name
        self.handoffs: dict[str, Agent] = {}
        if handoffs:
            for agent in handoffs:
                self._register_handoff(agent)

        # Lifecycle hooks
        self.hook_manager = HookManager()
        self._has_user_hooks: bool = bool(hooks)  # tracks explicitly-provided hooks only
        if hooks:
            for point, hook in hooks:
                self.hook_manager.add(point, hook)

        # Auto-attach memory persistence hooks when a MemoryStore is provided
        if memory is not None:
            self._attach_memory_persistence(memory)

    def _register_tool(self, t: Tool) -> None:
        """Add a tool, raising on duplicate names.

        Args:
            t: The tool to register.

        Raises:
            AgentError: If a tool with the same name is already registered.
        """
        if t.name in self.tools:
            raise AgentError(f"Duplicate tool name '{t.name}' on agent '{self.name}'")
        self.tools[t.name] = t

    def _register_handoff(self, agent: Agent) -> None:
        """Add a handoff target, raising on duplicate names.

        Args:
            agent: The target agent.

        Raises:
            AgentError: If a handoff with the same name is already registered.
        """
        if agent.name in self.handoffs:
            raise AgentError(f"Duplicate handoff agent '{agent.name}' on agent '{self.name}'")
        self.handoffs[agent.name] = agent

    def _attach_memory_persistence(self, memory: Any) -> None:
        """Auto-attach MemoryPersistence hooks if orbiter-memory is installed.

        Handles both ``AgentMemory`` (uses ``short_term`` store) and plain
        ``MemoryStore`` objects.  If the orbiter-memory package is not
        installed, this is a no-op.
        """
        try:
            from orbiter.memory.base import AgentMemory, MemoryStore  # pyright: ignore[reportMissingImports]
            from orbiter.memory.persistence import (  # pyright: ignore[reportMissingImports]
                MemoryPersistence,
            )
        except ImportError:
            return

        if isinstance(memory, AgentMemory):
            persistence = MemoryPersistence(memory.short_term)
            persistence.attach(self)
            self._memory_persistence = persistence
        elif isinstance(memory, MemoryStore):
            persistence = MemoryPersistence(memory)
            persistence.attach(self)
            self._memory_persistence = persistence

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools.

        Returns:
            A list of tool schema dicts suitable for LLM function calling.
        """
        return [t.to_schema() for t in self.tools.values()]

    def describe(self) -> dict[str, Any]:
        """Return a summary of the agent's capabilities.

        Useful for debugging, logging, and capability advertisement
        in multi-agent systems.

        Returns:
            A dict with the agent's name, model, tools, and configuration.
        """
        return {
            "name": self.name,
            "model": self.model,
            "tools": list(self.tools.keys()),
            "handoffs": list(self.handoffs.keys()),
            "max_steps": self.max_steps,
            "output_type": (self.output_type.__name__ if self.output_type else None),
        }

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        conversation_id: str | None = None,
    ) -> AgentOutput:
        """Execute the agent's LLM-tool loop with retry logic.

        Builds the message list, calls the LLM, and if tool calls are
        returned, executes them in parallel, feeds results back, and
        re-calls the LLM. The loop continues until a text-only response
        is produced or ``max_steps`` is reached.

        Args:
            input: User query string for this turn.
            messages: Prior conversation history.
            provider: An object with an ``async complete()`` method
                (e.g. a ``ModelProvider`` instance).
            max_retries: Maximum retry attempts for transient errors.
            conversation_id: Conversation scope override for this call only.
                When omitted, the agent's ``conversation_id`` attribute is
                used (auto-assigned UUID4 on first run if memory is set).

        Returns:
            Parsed ``AgentOutput`` from the final LLM response.

        Raises:
            AgentError: If no provider is supplied or all retries are exhausted.
        """
        if provider is None:
            raise AgentError(f"Agent '{self.name}' requires a provider for run()")

        # Resolve instructions (may be async callable)
        raw_instr = self.instructions
        if callable(raw_instr):
            if asyncio.iscoroutinefunction(raw_instr):
                instructions = await raw_instr(self.name)
            else:
                instructions = raw_instr(self.name)
        else:
            instructions = raw_instr

        # ---- Memory: load history and persist user input before LLM call ----
        history: list[Message] = list(messages) if messages else []
        if self._memory_persistence is not None:
            _active_conv = conversation_id or self.conversation_id
            if _active_conv is None:
                _active_conv = str(uuid.uuid4())
                if conversation_id is None:
                    self.conversation_id = _active_conv
            from orbiter.memory.base import HumanMemory, MemoryMetadata  # pyright: ignore[reportMissingImports]
            self._memory_persistence.metadata = MemoryMetadata(
                agent_id=self.name,
                task_id=_active_conv,
            )
            _db_history = await self._memory_persistence.load_history(
                agent_name=self.name,
                conversation_id=_active_conv,
                rounds=self.max_steps,
            )
            history = list(_db_history) + history
            await self._memory_persistence.store.add(
                HumanMemory(
                    content=input,
                    metadata=self._memory_persistence.metadata,
                )
            )
            _log.debug(
                "memory pre-run: agent=%s conversation=%s db_history=%d",
                self.name, _active_conv, len(_db_history),
            )
        # ---- end Memory ----

        # Build initial message list
        history.append(UserMessage(content=input))
        msg_list = build_messages(instructions, history)

        # ---- Context: apply windowing and summarization ----
        if self.context is not None:
            msg_list = await _apply_context_windowing(msg_list, self.context, provider)
        # ---- end Context ----

        # Tool schemas
        tool_schemas = self.get_tool_schemas() or None

        # Tool loop — iterate up to max_steps
        for _step in range(self.max_steps):
            output = await self._call_llm(msg_list, tool_schemas, provider, max_retries)

            # No tool calls — return the final text response
            if not output.tool_calls:
                return output

            # Execute tool calls and collect results
            actions = parse_tool_arguments(output.tool_calls)
            tool_results = await self._execute_tools(actions)

            # Append assistant message (with tool calls) and results to history
            msg_list.append(AssistantMessage(content=output.text, tool_calls=output.tool_calls))
            msg_list.extend(tool_results)

        # max_steps exhausted — return last output as-is
        return output

    async def _call_llm(
        self,
        msg_list: list[Message],
        tool_schemas: list[dict[str, Any]] | None,
        provider: Any,
        max_retries: int,
    ) -> AgentOutput:
        """Single LLM call with retry logic and lifecycle hooks."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                await self.hook_manager.run(HookPoint.PRE_LLM_CALL, agent=self, messages=msg_list)

                response = await provider.complete(
                    msg_list,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                await self.hook_manager.run(HookPoint.POST_LLM_CALL, agent=self, response=response)

                return parse_response(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    usage=response.usage,
                )

            except Exception as exc:
                if _is_context_length_error(exc):
                    _log.error("Context length exceeded on '%s'", self.name)
                    raise AgentError(
                        f"Context length exceeded on agent '{self.name}': {exc}"
                    ) from exc

                last_error = exc
                if attempt < max_retries - 1:
                    _log.warning(
                        "Retry %d/%d for '%s': %s", attempt + 1, max_retries, self.name, exc
                    )
                    delay = 2**attempt
                    await asyncio.sleep(delay)

        _log.error("Agent '%s' failed after %d retries", self.name, max_retries)
        raise AgentError(
            f"Agent '{self.name}' failed after {max_retries} retries: {last_error}"
        ) from last_error

    async def _execute_tools(
        self,
        actions: list[Any],
    ) -> list[ToolResult]:
        """Execute tool calls in parallel, catching errors per-tool."""
        results: list[ToolResult] = [
            ToolResult(tool_call_id="", tool_name="") for _ in range(len(actions))
        ]

        async def _run_one(idx: int) -> None:
            action = actions[idx]
            tool = self.tools.get(action.tool_name)

            # PRE_TOOL_CALL hook
            await self.hook_manager.run(
                HookPoint.PRE_TOOL_CALL,
                agent=self,
                tool_name=action.tool_name,
                arguments=action.arguments,
            )

            if tool is None:
                result = ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    error=f"Unknown tool '{action.tool_name}'",
                )
            else:
                try:
                    output = await tool.execute(**action.arguments)
                    content = output if isinstance(output, str) else str(output)
                    result = ToolResult(
                        tool_call_id=action.tool_call_id,
                        tool_name=action.tool_name,
                        content=content,
                    )
                except (ToolError, Exception) as exc:
                    _log.warning("Tool '%s' failed on '%s': %s", action.tool_name, self.name, exc)
                    result = ToolResult(
                        tool_call_id=action.tool_call_id,
                        tool_name=action.tool_name,
                        error=str(exc),
                    )

            # POST_TOOL_CALL hook
            await self.hook_manager.run(
                HookPoint.POST_TOOL_CALL,
                agent=self,
                tool_name=action.tool_name,
                result=result,
            )

            results[idx] = result

        async with asyncio.TaskGroup() as tg:
            for i in range(len(actions)):
                tg.create_task(_run_one(i))

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent configuration to a dict.

        Tools are serialized as importable dotted paths. Callable instructions,
        hooks, memory, and context cannot be serialized and will raise ValueError.

        Returns:
            A dict suitable for JSON serialization and later reconstruction
            via ``Agent.from_dict()``.

        Raises:
            ValueError: If the agent contains non-serializable components
                (callable instructions, hooks, closure-based tools, memory, context).
        """
        if callable(self.instructions):
            raise ValueError(
                f"Agent '{self.name}' has callable instructions which cannot be serialized. "
                "Use a string instruction instead."
            )
        if self.memory is not None and not self._memory_is_auto:
            raise ValueError(
                f"Agent '{self.name}' has a memory store which cannot be serialized."
            )
        if self._has_user_hooks:
            raise ValueError(
                f"Agent '{self.name}' has hooks which cannot be serialized."
            )
        if self.context is not None and not self._context_is_auto:
            raise ValueError(
                f"Agent '{self.name}' has a context engine which cannot be serialized."
            )

        data: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "instructions": self.instructions,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Serialize tools as importable dotted paths
        if self.tools:
            data["tools"] = [_serialize_tool(t) for t in self.tools.values()]

        # Serialize handoffs recursively
        if self.handoffs:
            data["handoffs"] = [agent.to_dict() for agent in self.handoffs.values()]

        # Serialize output_type as importable dotted path
        if self.output_type is not None:
            data["output_type"] = f"{self.output_type.__module__}.{self.output_type.__qualname__}"

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Reconstruct an Agent from a dict produced by ``to_dict()``.

        Tools are resolved by importing dotted paths. Handoff agents are
        reconstructed recursively.

        Args:
            data: Dict as produced by ``Agent.to_dict()``.

        Returns:
            A reconstructed ``Agent`` instance.

        Raises:
            ValueError: If a tool or output_type path cannot be imported.
        """
        tools: list[Tool] | None = None
        if "tools" in data:
            tools = [_deserialize_tool(t) for t in data["tools"]]

        handoffs: list[Agent] | None = None
        if "handoffs" in data:
            handoffs = [Agent.from_dict(h) for h in data["handoffs"]]

        output_type: type[BaseModel] | None = None
        if "output_type" in data:
            output_type = _import_object(data["output_type"])

        return cls(
            name=data["name"],
            model=data.get("model", "openai:gpt-4o"),
            instructions=data.get("instructions", ""),
            tools=tools,
            handoffs=handoffs,
            output_type=output_type,
            max_steps=data.get("max_steps", 10),
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens"),
        )

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"model={self.model!r}"]
        if self.tools:
            parts.append(f"tools={list(self.tools.keys())}")
        if self.handoffs:
            parts.append(f"handoffs={list(self.handoffs.keys())}")
        return f"Agent({', '.join(parts)})"


def _is_context_length_error(exc: Exception) -> bool:
    """Check if an exception represents a context-length overflow.

    Detects errors with a ``code`` attribute of ``"context_length"``
    (set by ``ModelError``) or common provider error messages.
    """
    code = getattr(exc, "code", "")
    if code == "context_length":
        return True
    msg = str(exc).lower()
    return "context_length" in msg or "context length" in msg


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_tool(t: Tool) -> str | dict[str, Any]:
    """Serialize a tool to an importable dotted path or a dict.

    For ``MCPToolWrapper``, returns a dict with an ``__mcp_tool__`` marker.
    For ``FunctionTool``, uses the wrapped function's module and qualname.
    For custom ``Tool`` subclasses, uses the class's module and qualname.

    Raises:
        ValueError: If the tool cannot be serialized (e.g., closures, lambdas).
    """
    # MCPToolWrapper — serialize as a dict with server config
    try:
        from orbiter.mcp.tools import MCPToolWrapper  # pyright: ignore[reportMissingImports]

        if isinstance(t, MCPToolWrapper):
            return t.to_dict()
    except ImportError:
        pass

    from orbiter.tool import FunctionTool

    if isinstance(t, FunctionTool):
        fn = t._fn
        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", None)
        if not module or not qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a function without __module__ or __qualname__ "
                "and cannot be serialized."
            )
        # Detect closures/lambdas (qualname contains '<')
        if "<" in qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a closure or lambda ({qualname}) "
                "which cannot be serialized. Use a module-level function instead."
            )
        return f"{module}.{qualname}"

    # Custom Tool subclass — serialize the class itself
    cls = type(t)
    module = cls.__module__
    qualname = cls.__qualname__
    if "<" in qualname:
        raise ValueError(
            f"Tool '{t.name}' is a locally-defined class ({qualname}) "
            "which cannot be serialized."
        )
    return f"{module}.{qualname}"


def _deserialize_tool(path: str | dict[str, Any]) -> Tool:
    """Deserialize a tool from an importable dotted path or a dict.

    If ``path`` is a dict with an ``__mcp_tool__`` marker, reconstructs an
    ``MCPToolWrapper`` via ``from_dict()``.

    If the imported object is a callable (function), wraps it as a FunctionTool.
    If it's already a Tool instance, returns it directly.
    If it's a Tool subclass, instantiates it.

    Raises:
        ValueError: If the path cannot be imported or doesn't resolve to a tool.
    """
    if isinstance(path, dict):
        if path.get("__mcp_tool__"):
            from orbiter.mcp.tools import (  # pyright: ignore[reportMissingImports]
                MCPToolWrapper,
            )

            return MCPToolWrapper.from_dict(path)
        raise ValueError(f"Unknown tool dict format: {path!r}")

    from orbiter.tool import FunctionTool

    obj = _import_object(path)

    # Already a Tool instance (e.g., @tool decorated at module level)
    if isinstance(obj, Tool):
        return obj

    # A Tool subclass — instantiate it
    if isinstance(obj, type) and issubclass(obj, Tool):
        return obj()

    # A plain callable — wrap it
    if callable(obj):
        return FunctionTool(obj)

    raise ValueError(
        f"Imported '{path}' is not a callable or Tool instance: {type(obj)}"
    )


def _import_object(dotted_path: str) -> Any:
    """Import an object from a dotted path like 'package.module.ClassName'.

    Tries progressively shorter module paths, resolving the remainder
    via getattr.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid dotted path: {dotted_path!r}")

    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        pass

    # Try splitting further for nested attributes (e.g., module.Class.method)
    parts = dotted_path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ValueError(f"Cannot import '{dotted_path}'")
