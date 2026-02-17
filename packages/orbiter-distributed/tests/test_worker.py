"""Tests for Worker task execution lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orbiter.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from orbiter.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)
from orbiter.distributed.worker import (  # pyright: ignore[reportMissingImports]
    Worker,
    _generate_worker_id,
)

# ---------------------------------------------------------------------------
# _generate_worker_id
# ---------------------------------------------------------------------------


class TestGenerateWorkerId:
    def test_format(self) -> None:
        wid = _generate_worker_id()
        parts = wid.split("-")
        # hostname-pid-suffix  (hostname may contain dashes, so at least 3 parts)
        assert len(parts) >= 3

    def test_unique(self) -> None:
        ids = {_generate_worker_id() for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# Worker.__init__
# ---------------------------------------------------------------------------


class TestWorkerInit:
    def test_defaults(self) -> None:
        w = Worker("redis://localhost:6379")
        assert w._redis_url == "redis://localhost:6379"
        assert w._concurrency == 1
        assert w._queue_name == "orbiter:tasks"
        assert w._heartbeat_ttl == 30
        assert w._worker_id  # auto-generated
        assert w.tasks_processed == 0
        assert w.tasks_failed == 0

    def test_custom_params(self) -> None:
        w = Worker(
            "redis://host:1234",
            worker_id="my-worker",
            concurrency=4,
            queue_name="custom:q",
            heartbeat_ttl=60,
        )
        assert w.worker_id == "my-worker"
        assert w._concurrency == 4
        assert w._queue_name == "custom:q"
        assert w._heartbeat_ttl == 60

    def test_auto_worker_id(self) -> None:
        w1 = Worker("redis://localhost")
        w2 = Worker("redis://localhost")
        assert w1.worker_id != w2.worker_id


# ---------------------------------------------------------------------------
# Worker._reconstruct_agent
# ---------------------------------------------------------------------------


class TestWorkerReconstructAgent:
    def test_agent_config(self) -> None:
        w = Worker("redis://localhost")
        config = {"name": "test-agent", "model": "openai:gpt-4o"}
        with patch("orbiter.agent.Agent") as mock_agent_cls:
            mock_agent_cls.from_dict.return_value = "mock-agent"
            result = w._reconstruct_agent(config)
            mock_agent_cls.from_dict.assert_called_once_with(config)
            assert result == "mock-agent"

    def test_swarm_config(self) -> None:
        w = Worker("redis://localhost")
        config = {
            "agents": [{"name": "a1", "model": "openai:gpt-4o"}],
            "mode": "workflow",
        }
        with patch("orbiter.swarm.Swarm") as mock_swarm_cls:
            mock_swarm_cls.from_dict.return_value = "mock-swarm"
            result = w._reconstruct_agent(config)
            mock_swarm_cls.from_dict.assert_called_once_with(config)
            assert result == "mock-swarm"


# ---------------------------------------------------------------------------
# Worker._execute_task
# ---------------------------------------------------------------------------


class TestWorkerExecuteTask:
    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=False,
        )

        async def _fake_run_agent(
            agent: object, t: TaskPayload, token: CancellationToken
        ) -> str:
            return "Hello!"

        # Mock the agent reconstruction and streaming
        mock_agent = MagicMock()
        with (
            patch.object(w, "_reconstruct_agent", return_value=mock_agent),
            patch.object(w, "_run_agent", side_effect=_fake_run_agent),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Verify status transitions
        calls = w._store.set_status.call_args_list
        assert calls[0].args[1] == TaskStatus.RUNNING
        assert calls[1].args[1] == TaskStatus.COMPLETED
        assert calls[1].kwargs["result"] == {"output": "Hello!"}

        # Verify ack
        w._broker.ack.assert_called_once_with("task-1")
        assert w.tasks_processed == 1
        assert w.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_failed_execution_with_retries(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-2", status=TaskStatus.RUNNING, retries=0
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-2",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=ValueError("bad config")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should set FAILED then RETRYING
        status_calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in status_calls]
        assert TaskStatus.RUNNING in statuses
        assert TaskStatus.FAILED in statuses
        assert TaskStatus.RETRYING in statuses

        # Should nack (re-queue) since retries < max
        w._broker.nack.assert_called_once_with("task-2")
        assert w.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_failed_execution_max_retries_exhausted(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._broker.max_retries = 3
        w._store = AsyncMock()
        w._store.get_status.return_value = TaskResult(
            task_id="task-3", status=TaskStatus.RUNNING, retries=3
        )
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-3",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="fail",
        )

        with (
            patch.object(w, "_reconstruct_agent", side_effect=RuntimeError("crash")),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should ack (not nack) since retries exhausted
        w._broker.ack.assert_called_once_with("task-3")
        w._broker.nack.assert_not_called()
        assert w.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_current_task_id_tracked(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-4",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        captured_task_id: str | None = None

        async def capture_run(
            agent: object, t: TaskPayload, token: CancellationToken
        ) -> str:
            nonlocal captured_task_id
            captured_task_id = w._current_task_id
            return "done"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=capture_run),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        assert captured_task_id == "task-4"
        assert w._current_task_id is None  # cleared after execution


# ---------------------------------------------------------------------------
# Worker._run_agent
# ---------------------------------------------------------------------------


class TestWorkerRunAgent:
    @pytest.mark.asyncio
    async def test_streams_and_publishes_events(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-5",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=True,
        )

        from orbiter.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        events = [
            TextEvent(text="Hello", agent_name="agent"),
            TextEvent(text=" world", agent_name="agent"),
            StatusEvent(status="completed", agent_name="agent", message="done"),
        ]

        mock_agent = MagicMock()

        # Patch orbiter.runner.run (the actual module where run is defined)
        # so the local import inside _run_agent picks up our mock
        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("orbiter.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "Hello world"
        assert w._publisher.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_collects_only_text_events(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-6",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from orbiter.types import (  # pyright: ignore[reportMissingImports]
            StatusEvent,
            TextEvent,
            ToolCallEvent,
        )

        events = [
            ToolCallEvent(tool_name="search", tool_call_id="tc-1", agent_name="agent"),
            TextEvent(text="result", agent_name="agent"),
            StatusEvent(status="completed", agent_name="agent", message="done"),
        ]

        mock_agent = MagicMock()

        mock_run = MagicMock()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = _fake_stream_gen

        token = CancellationToken()
        with patch("orbiter.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        # Only TextEvent text should be collected
        assert result == "result"


# ---------------------------------------------------------------------------
# Worker._claim_loop
# ---------------------------------------------------------------------------


class TestWorkerClaimLoop:
    @pytest.mark.asyncio
    async def test_loops_until_shutdown(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()

        call_count = 0

        async def fake_claim(worker_id: str, *, timeout: float = 2.0) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                w._shutdown_event.set()
            return None

        w._broker.claim = fake_claim  # type: ignore[assignment]
        await w._claim_loop()
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_executes_claimed_task(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()

        task = TaskPayload(
            task_id="task-7",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        calls = 0

        async def fake_claim(worker_id: str, *, timeout: float = 2.0) -> TaskPayload | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                return task
            w._shutdown_event.set()
            return None

        w._broker.claim = fake_claim  # type: ignore[assignment]

        with patch.object(w, "_execute_task", new_callable=AsyncMock) as mock_exec:
            await w._claim_loop()
            mock_exec.assert_called_once_with(task)


# ---------------------------------------------------------------------------
# Worker.stop
# ---------------------------------------------------------------------------


class TestWorkerStop:
    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_event(self) -> None:
        w = Worker("redis://localhost")
        assert not w._shutdown_event.is_set()
        await w.stop()
        assert w._shutdown_event.is_set()


# ---------------------------------------------------------------------------
# Worker.start integration
# ---------------------------------------------------------------------------


class TestWorkerStart:
    @pytest.mark.asyncio
    async def test_start_connects_and_runs(self) -> None:
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        # Immediately signal shutdown so start() exits
        w._shutdown_event.set()

        with (
            patch.object(w, "_claim_loop", new_callable=AsyncMock),
            patch.object(w, "_heartbeat_loop", new_callable=AsyncMock),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value = MagicMock()
            mock_loop.return_value.add_signal_handler = MagicMock()
            await w.start()

        w._broker.connect.assert_called_once()
        w._store.connect.assert_called_once()
        w._publisher.connect.assert_called_once()
        w._broker.disconnect.assert_called_once()
        w._store.disconnect.assert_called_once()
        w._publisher.disconnect.assert_called_once()
