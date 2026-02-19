"""Integration tests: Context + PromptBuilder + Processor + Workspace end-to-end."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from orbiter.context import (  # pyright: ignore[reportMissingImports]
    ArtifactType,
    AutomationMode,
    Checkpoint,
    CheckpointStore,
    Context,
    ContextConfig,
    ContextError,
    ContextProcessor,
    ContextState,
    Neuron,
    ProcessorPipeline,
    PromptBuilder,
    SummarizeProcessor,
    TokenTracker,
    ToolResultOffloader,
    Workspace,
    get_context_tools,
    get_file_tools,
    get_knowledge_tools,
    get_planning_tools,
    make_config,
    neuron_registry,
)
from orbiter.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
    KnowledgeStore,
)
from orbiter.context.tools import (  # pyright: ignore[reportMissingImports]
    planning_tool_add,
    planning_tool_get,
)

# ── Public API import tests ──────────────────────────────────────────


class TestPublicAPIImports:
    """Verify all expected names are importable from orbiter.context."""

    def test_core_classes(self) -> None:
        assert Context is not None
        assert ContextConfig is not None
        assert ContextState is not None
        assert ContextError is not None

    def test_prompt_building(self) -> None:
        assert PromptBuilder is not None
        assert Neuron is not None
        assert neuron_registry is not None

    def test_processor_pipeline(self) -> None:
        assert ContextProcessor is not None
        assert ProcessorPipeline is not None
        assert SummarizeProcessor is not None
        assert ToolResultOffloader is not None

    def test_workspace_and_artifacts(self) -> None:
        assert Workspace is not None
        assert ArtifactType is not None

    def test_checkpoint(self) -> None:
        assert Checkpoint is not None
        assert CheckpointStore is not None

    def test_config_and_mode(self) -> None:
        assert AutomationMode is not None
        assert make_config is not None

    def test_token_tracker(self) -> None:
        assert TokenTracker is not None

    def test_tool_factories(self) -> None:
        assert get_context_tools is not None
        assert get_planning_tools is not None
        assert get_knowledge_tools is not None
        assert get_file_tools is not None


# ── Context + PromptBuilder end-to-end ───────────────────────────────


class TestContextPromptBuilderE2E:
    """End-to-end: Context creation -> neuron registration -> prompt building."""

    async def test_basic_prompt_building(self) -> None:
        ctx = Context("task-1")
        ctx.state.set("task_input", "Summarize this document")

        builder = PromptBuilder(ctx)
        builder.add("system")
        builder.add("task")
        prompt = await builder.build()

        assert "task-1" in prompt
        assert "Summarize this document" in prompt

    async def test_prompt_with_history(self) -> None:
        ctx = Context("task-2")
        ctx.state.set(
            "history",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        )

        builder = PromptBuilder(ctx)
        builder.add("history")
        prompt = await builder.build()

        assert "Hello" in prompt
        assert "Hi there!" in prompt

    async def test_prompt_with_todos(self) -> None:
        ctx = Context("task-3")
        ctx.state.set(
            "todos",
            [
                {"item": "Research topic", "done": True},
                {"item": "Write draft", "done": False},
            ],
        )

        builder = PromptBuilder(ctx)
        builder.add("todo")
        prompt = await builder.build()

        assert "Research topic" in prompt
        assert "Write draft" in prompt

    async def test_full_prompt_composition(self) -> None:
        ctx = Context("task-4")
        ctx.state.set("task_id", "task-4")
        ctx.state.set("input", "Build a web app")
        ctx.state.set("todos", [{"item": "Setup", "done": False}])
        ctx.state.set("facts", ["Python 3.11+", "UV package manager"])

        builder = PromptBuilder(ctx)
        builder.add("task").add("todo").add("fact").add("system")
        prompt = await builder.build()

        # task neuron is priority 1, todo is 2, fact is 50, system is 100
        parts = prompt.split("\n\n")
        assert len(parts) >= 4

        # All content present
        assert "task-4" in prompt
        assert "Setup" in prompt
        assert "Python 3.11+" in prompt


# ── Context + Processor pipeline end-to-end ──────────────────────────


class TestContextProcessorE2E:
    """End-to-end: Context + ProcessorPipeline with built-in processors."""

    async def test_summarize_processor_triggers(self) -> None:
        config = make_config("copilot", summary_threshold=3)
        ctx = Context("task-5", config=config)
        ctx.state.set("history", [{"role": "user", "content": f"msg-{i}"} for i in range(5)])

        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())

        await pipeline.fire("pre_llm_call", ctx)

        assert ctx.state.get("needs_summary") is True
        candidates = ctx.state.get("summary_candidates")
        assert candidates is not None
        assert len(candidates) == 2  # 5 - 3 = 2 excess

    async def test_tool_result_offloader(self) -> None:
        ctx = Context("task-6")
        pipeline = ProcessorPipeline()
        pipeline.register(ToolResultOffloader(max_size=50))

        large_result = "x" * 200
        payload: dict[str, Any] = {
            "tool_result": large_result,
            "tool_name": "search",
            "tool_call_id": "tc-1",
        }
        await pipeline.fire("post_tool_call", ctx, payload)

        offloaded = ctx.state.get("offloaded_results")
        assert offloaded is not None
        assert len(offloaded) == 1
        assert offloaded[0]["tool_name"] == "search"
        assert offloaded[0]["size"] == 200

        # Payload was mutated with truncated reference
        assert "[Result truncated" in payload["tool_result"]

    async def test_multi_processor_pipeline(self) -> None:
        """Multiple processors fire in sequence for different events."""
        config = make_config("copilot", summary_threshold=2)
        ctx = Context("task-7", config=config)
        ctx.state.set("history", [{"role": "user", "content": "msg"}] * 5)

        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        pipeline.register(ToolResultOffloader(max_size=10))

        # Fire pre_llm_call — SummarizeProcessor triggers
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("needs_summary") is True

        # Fire post_tool_call — ToolResultOffloader triggers
        payload: dict[str, Any] = {"tool_result": "a" * 100, "tool_name": "t"}
        await pipeline.fire("post_tool_call", ctx, payload)
        assert ctx.state.get("offloaded_results") is not None


# ── Context + Workspace end-to-end ──────────────────────────────────


class TestContextWorkspaceE2E:
    """End-to-end: Context + Workspace with knowledge store integration."""

    async def test_workspace_write_and_knowledge_search(self) -> None:
        ks = KnowledgeStore()
        workspace = Workspace("ws-1", knowledge_store=ks)

        await workspace.write("readme", "Orbiter is a multi-agent framework for LLMs")
        await workspace.write("guide", "Quick start: install with pip install orbiter")

        # Knowledge store was auto-indexed
        results = ks.search("multi-agent framework")
        assert len(results) > 0
        assert any("Orbiter" in r.chunk.content for r in results)

    async def test_workspace_versioning_with_context(self) -> None:
        ctx = Context("task-8")
        workspace = Workspace("ws-2")

        await workspace.write("notes", "Version 1")
        await workspace.write("notes", "Version 2")
        await workspace.write("notes", "Version 3")

        # Store workspace in context state
        ctx.state.set("workspace", workspace)

        # Access via context
        ws = ctx.state.get("workspace")
        assert ws.read("notes") == "Version 3"
        versions = ws.version_history("notes")
        assert len(versions) == 3

    async def test_workspace_with_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace("ws-3", storage_path=tmp)
            await workspace.write("data.json", '{"key": "value"}', artifact_type=ArtifactType.JSON)

            # Verify persisted to filesystem
            content_path = Path(tmp) / "data.json" / "content"
            assert content_path.exists()
            assert content_path.read_text() == '{"key": "value"}'


# ── Full lifecycle: Context + PromptBuilder + Processor + Workspace ──


class TestFullLifecycle:
    """End-to-end lifecycle: create -> populate -> process -> build prompt -> checkpoint."""

    async def test_full_agent_context_lifecycle(self) -> None:
        # 1. Create context with navigator mode
        config = make_config("navigator", summary_threshold=3)
        ctx = Context("main-task", config=config)

        # 2. Set up workspace + knowledge store
        ks = KnowledgeStore()
        workspace = Workspace("ws-main", knowledge_store=ks)
        ctx.state.set("workspace", workspace)
        ctx.state.set("knowledge_store", ks)

        # 3. Write artifacts
        await workspace.write("spec", "Build a REST API with authentication and rate limiting")
        await workspace.write("notes", "Use FastAPI with JWT tokens for auth")

        # 4. Set task state
        ctx.state.set("task_input", "Implement the authentication module")
        ctx.state.set(
            "todos",
            [
                {"item": "Design auth schema", "done": True},
                {"item": "Implement JWT handler", "done": False},
                {"item": "Write tests", "done": False},
            ],
        )

        # 5. Build prompt
        builder = PromptBuilder(ctx)
        builder.add("task").add("todo").add("system")
        prompt = await builder.build()

        assert "main-task" in prompt
        assert "Implement the authentication module" in prompt
        assert "JWT handler" in prompt

        # 6. Track tokens
        ctx.add_tokens({"prompt_tokens": 500, "output_tokens": 150})

        # 7. Set up processor pipeline
        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())

        # 8. Simulate history growth past threshold
        ctx.state.set("history", [{"role": "user", "content": f"Turn {i}"} for i in range(5)])
        await pipeline.fire("pre_llm_call", ctx)
        assert ctx.state.get("needs_summary") is True

        # 9. Checkpoint
        cp = ctx.snapshot(metadata={"step": 1})
        assert cp.task_id == "main-task"
        assert cp.values["needs_summary"] is True
        assert cp.token_usage["prompt_tokens"] == 500

        # 10. Restore from checkpoint
        restored = Context.restore(cp, config=config)
        assert restored.state.get("needs_summary") is True
        assert restored.token_usage["prompt_tokens"] == 500

    async def test_fork_merge_with_workspace(self) -> None:
        """Fork a child context, work in it, merge back."""
        # Parent context with workspace
        ks = KnowledgeStore()
        workspace = Workspace("ws-fork", knowledge_store=ks)

        parent = Context("parent-task")
        parent.state.set("workspace", workspace)
        parent.state.set("result_count", 0)
        parent.add_tokens({"prompt_tokens": 100})

        # Fork for subtask
        child = parent.fork("subtask-1")
        assert child.state.get("workspace") is workspace  # Inherited

        # Child does work
        await workspace.write("subtask-output", "Subtask result data")
        child.state.set("result_count", 1)
        child.state.set("subtask_done", True)
        child.add_tokens({"prompt_tokens": 200, "output_tokens": 50})

        # Merge back
        parent.merge(child)
        assert parent.state.get("subtask_done") is True
        assert parent.state.get("result_count") == 1  # Overwritten by child local
        # Net token delta: child started with parent's 100, added 200 => net = 200
        assert parent.token_usage["prompt_tokens"] == 300  # 100 + 200 net
        assert parent.token_usage["output_tokens"] == 50

    async def test_context_tools_with_workspace(self) -> None:
        """Context tools work with workspace via context state."""
        ctx = Context("tool-task")
        ks = KnowledgeStore()
        workspace = Workspace("ws-tools", knowledge_store=ks)
        ctx.state.set("workspace", workspace)
        ctx.state.set("knowledge_store", ks)

        # Write an artifact via workspace
        await workspace.write("readme", "# My Project\nA framework for building agents")

        # Use planning tool
        planning_tool_add.bind(ctx)
        result = await planning_tool_add.execute(item="Read the readme")
        assert "Added todo" in result

        # Use planning tool get
        planning_tool_get.bind(ctx)
        result = await planning_tool_get.execute()
        assert "Read the readme" in result

        # Check state was mutated
        todos = ctx.state.get("todos")
        assert len(todos) == 1
        assert todos[0]["item"] == "Read the readme"


# ── Custom processor integration ─────────────────────────────────────


class _CounterProcessor(ContextProcessor):
    """Test processor that counts invocations."""

    def __init__(self, event: str) -> None:
        super().__init__(event, name="counter")
        self.count = 0

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        self.count += 1
        ctx.state.set("counter", self.count)


class TestCustomProcessorIntegration:
    """Custom processors integrate with context and pipeline."""

    async def test_custom_processor_mutates_state(self) -> None:
        ctx = Context("custom-task")
        proc = _CounterProcessor("on_step")

        pipeline = ProcessorPipeline()
        pipeline.register(proc)

        await pipeline.fire("on_step", ctx)
        await pipeline.fire("on_step", ctx)
        await pipeline.fire("on_step", ctx)

        assert ctx.state.get("counter") == 3
        assert proc.count == 3

    async def test_mixed_processors(self) -> None:
        """Mix custom and built-in processors."""
        config = make_config("copilot", summary_threshold=2)
        ctx = Context("mixed-task", config=config)
        ctx.state.set("history", [{"role": "user", "content": "msg"}] * 5)

        counter = _CounterProcessor("pre_llm_call")
        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        pipeline.register(counter)

        await pipeline.fire("pre_llm_call", ctx)

        assert ctx.state.get("needs_summary") is True
        assert ctx.state.get("counter") == 1


# ── Agent context wiring ─────────────────────────────────────────────


class TestAgentContextWiring:
    """Verify Agent accepts and stores context."""

    def test_agent_accepts_context(self) -> None:
        from orbiter.agent import Agent

        ctx = Context("agent-task")
        agent = Agent(name="test", context=ctx)
        assert agent.context is ctx

    def test_agent_context_default_auto_created(self) -> None:
        """Agent without explicit context auto-creates ContextConfig(mode='copilot')."""
        from orbiter.agent import Agent
        from orbiter.context.config import AutomationMode, ContextConfig  # pyright: ignore[reportMissingImports]

        agent = Agent(name="test")
        assert isinstance(agent.context, ContextConfig)
        assert agent.context.mode == AutomationMode.COPILOT
        assert agent._context_is_auto is True

    def test_agent_describe_does_not_include_context(self) -> None:
        from orbiter.agent import Agent

        ctx = Context("agent-task")
        agent = Agent(name="test", context=ctx)
        desc = agent.describe()
        # describe() focuses on capabilities, not internal state
        assert "name" in desc
        assert "model" in desc
