"""Distributed worker process — claims tasks from the queue and executes agents."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import signal
import socket
import time
from typing import TYPE_CHECKING, Any, Literal

import redis.asyncio as aioredis

from orbiter.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from orbiter.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from orbiter.distributed.events import EventPublisher  # pyright: ignore[reportMissingImports]
from orbiter.distributed.metrics import (  # pyright: ignore[reportMissingImports]
    record_task_cancelled,
    record_task_completed,
    record_task_failed,
)
from orbiter.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from orbiter.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from orbiter.distributed.temporal import HAS_TEMPORAL  # pyright: ignore[reportMissingImports]
from orbiter.observability.propagation import (  # pyright: ignore[reportMissingImports]
    BaggagePropagator,
    DictCarrier,
)
from orbiter.observability.tracing import aspan  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from orbiter.distributed.temporal import (  # pyright: ignore[reportMissingImports]
        TemporalExecutor,
    )


def _generate_worker_id() -> str:
    """Generate a unique worker ID from hostname, PID, and a random suffix."""
    hostname = socket.gethostname()
    pid = os.getpid()
    suffix = random.randbytes(4).hex()
    return f"{hostname}-{pid}-{suffix}"


def _deserialize_messages(raw: list[dict[str, Any]]) -> list[Any]:
    """Convert a list of message dicts to typed Message objects.

    Dispatches on the ``role`` field to create the appropriate Pydantic model.

    Args:
        raw: List of message dicts, each with a ``role`` key.

    Returns:
        List of typed Message objects (UserMessage, AssistantMessage, etc.).

    Raises:
        ValueError: If a message has an unknown role.
    """
    from orbiter.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolResult,
        UserMessage,
    )

    _role_map = {
        "user": UserMessage,
        "assistant": AssistantMessage,
        "system": SystemMessage,
        "tool": ToolResult,
    }

    messages: list[Any] = []
    for msg in raw:
        role = msg.get("role", "")
        cls = _role_map.get(role)
        if cls is None:
            raise ValueError(f"Unknown message role: {role!r}")
        messages.append(cls(**msg))
    return messages


class Worker:
    """Claims tasks from a Redis queue, executes agents, and publishes events.

    The worker:
    1. Claims tasks from the queue via :class:`TaskBroker`.
    2. Reconstructs the agent from ``agent_config`` via ``Agent.from_dict()``.
    3. Runs ``run.stream()`` with ``detailed=task.detailed``.
    4. Publishes each streaming event via :class:`EventPublisher`.
    5. Updates task status in :class:`TaskStore`.
    6. Publishes a heartbeat to Redis periodically.

    Args:
        redis_url: Redis connection URL.
        worker_id: Unique worker identifier. Auto-generated if not provided.
        concurrency: Number of concurrent task executions (default 1).
        queue_name: Redis Streams queue name (default ``"orbiter:tasks"``).
        heartbeat_ttl: TTL in seconds for the heartbeat key (default 30).
        executor: Execution backend — ``"local"`` for direct execution or
            ``"temporal"`` for durable Temporal workflows.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        worker_id: str | None = None,
        concurrency: int = 1,
        queue_name: str = "orbiter:tasks",
        heartbeat_ttl: int = 30,
        executor: Literal["local", "temporal"] = "local",
    ) -> None:
        self._redis_url = redis_url
        self._worker_id = worker_id or _generate_worker_id()
        self._concurrency = concurrency
        self._queue_name = queue_name
        self._heartbeat_ttl = heartbeat_ttl
        self._executor_type = executor

        self._broker = TaskBroker(redis_url, queue_name=queue_name)
        self._store = TaskStore(redis_url)
        self._publisher = EventPublisher(redis_url)
        self._temporal_executor: TemporalExecutor | None = None

        if executor == "temporal":
            if not HAS_TEMPORAL:
                msg = (
                    "Temporal executor requires temporalio to be installed. "
                    "Install it with: pip install orbiter-distributed[temporal]"
                )
                raise ImportError(msg)
            from orbiter.distributed.temporal import (  # pyright: ignore[reportMissingImports]
                TemporalExecutor,
            )

            self._temporal_executor = TemporalExecutor()

        self._shutdown_event = asyncio.Event()
        self._tasks_processed = 0
        self._tasks_failed = 0
        self._current_task_id: str | None = None
        self._started_at: float = 0.0

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def tasks_processed(self) -> int:
        return self._tasks_processed

    @property
    def tasks_failed(self) -> int:
        return self._tasks_failed

    async def start(self) -> None:
        """Enter the claim-execute loop until shutdown is signalled.

        Registers SIGINT/SIGTERM handlers for graceful shutdown.
        When ``executor="temporal"``, also connects the Temporal executor.
        """
        self._started_at = time.time()

        await self._broker.connect()
        await self._store.connect()
        await self._publisher.connect()

        if self._temporal_executor is not None:
            await self._temporal_executor.connect()

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal)

        try:
            # Start heartbeat in the background
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Run claim-execute workers
            workers = [
                asyncio.create_task(self._claim_loop())
                for _ in range(self._concurrency)
            ]
            await asyncio.gather(*workers)
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        finally:
            if self._temporal_executor is not None:
                await self._temporal_executor.disconnect()
            await self._broker.disconnect()
            await self._store.disconnect()
            await self._publisher.disconnect()

    async def stop(self) -> None:
        """Signal the worker to shut down gracefully."""
        self._shutdown_event.set()

    def _handle_signal(self) -> None:
        """Signal handler for SIGINT/SIGTERM."""
        self._shutdown_event.set()

    async def _heartbeat_loop(self) -> None:
        """Publish a heartbeat to Redis periodically."""
        r: aioredis.Redis = aioredis.from_url(
            self._redis_url, decode_responses=True
        )
        key = f"orbiter:workers:{self._worker_id}"
        try:
            while not self._shutdown_event.is_set():
                fields = {
                    "status": "running",
                    "tasks_processed": str(self._tasks_processed),
                    "tasks_failed": str(self._tasks_failed),
                    "current_task_id": self._current_task_id or "",
                    "started_at": str(self._started_at),
                    "last_heartbeat": str(time.time()),
                    "concurrency": str(self._concurrency),
                    "hostname": socket.gethostname(),
                }
                await r.hset(key, mapping=fields)  # type: ignore[misc]
                await r.expire(key, self._heartbeat_ttl)  # type: ignore[misc]
                await asyncio.sleep(self._heartbeat_ttl / 3)
        finally:
            await r.aclose()

    async def _claim_loop(self) -> None:
        """Claim tasks and execute them until shutdown."""
        while not self._shutdown_event.is_set():
            task = await self._broker.claim(self._worker_id, timeout=2.0)
            if task is None:
                continue
            await self._execute_task(task)

    async def _execute_task(self, task: TaskPayload) -> None:
        """Execute a single task: reconstruct agent, stream, update status."""
        self._current_task_id = task.task_id
        token = CancellationToken()

        # Extract trace context from task metadata (injected by client)
        trace_context = task.metadata.get("trace_context")
        if isinstance(trace_context, dict):
            carrier = DictCarrier(trace_context)
            propagator = BaggagePropagator()
            propagator.extract(carrier)

        # Start listening for cancel signals on orbiter:cancel:{task_id}
        cancel_task = asyncio.create_task(
            self._listen_for_cancel(task.task_id, token)
        )

        started_at = time.time()
        wait_time = started_at - task.created_at if task.created_at > 0 else 0.0

        async with aspan(
            "orbiter.distributed.execute",
            attributes={
                "dist.task_id": task.task_id,
                "dist.worker_id": self._worker_id,
            },
        ) as s:
            try:
                # Mark as RUNNING
                await self._store.set_status(
                    task.task_id,
                    TaskStatus.RUNNING,
                    worker_id=self._worker_id,
                    started_at=started_at,
                )

                if self._temporal_executor is not None:
                    # Delegate to Temporal for durable execution
                    result_text = await self._temporal_executor.execute_task(
                        task,
                        self._store,
                        self._publisher,
                        token,
                        self._worker_id,
                    )
                else:
                    # Local execution: reconstruct agent and stream directly
                    agent = self._reconstruct_agent(task.agent_config)
                    result_text = await self._run_agent(agent, task, token)

                duration = time.time() - started_at

                if token.cancelled:
                    # Cancellation took effect during execution
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.CANCELLED,
                        completed_at=time.time(),
                    )
                    await self._broker.ack(task.task_id)
                    record_task_cancelled(
                        task_id=task.task_id,
                        worker_id=self._worker_id,
                    )
                else:
                    # Mark as COMPLETED
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.COMPLETED,
                        completed_at=time.time(),
                        result={"output": result_text},
                    )
                    await self._broker.ack(task.task_id)
                    self._tasks_processed += 1
                    record_task_completed(
                        task_id=task.task_id,
                        worker_id=self._worker_id,
                        duration=duration,
                        wait_time=wait_time,
                    )

            except Exception as exc:
                s.record_exception(exc)
                self._tasks_failed += 1
                duration = time.time() - started_at
                await self._store.set_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    completed_at=time.time(),
                    error=str(exc),
                )
                record_task_failed(
                    task_id=task.task_id,
                    worker_id=self._worker_id,
                    duration=duration,
                )

                # Check if retries remain
                status = await self._store.get_status(task.task_id)
                retries = status.retries if status else 0
                if retries < self._broker.max_retries:
                    await self._store.set_status(
                        task.task_id,
                        TaskStatus.RETRYING,
                        retries=retries + 1,
                    )
                    await self._broker.nack(task.task_id)
                else:
                    await self._broker.ack(task.task_id)

            finally:
                self._current_task_id = None
                cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_task

    async def _listen_for_cancel(
        self, task_id: str, token: CancellationToken
    ) -> None:
        """Subscribe to ``orbiter:cancel:{task_id}`` and set the token on signal."""
        r: aioredis.Redis = aioredis.from_url(
            self._redis_url, decode_responses=True
        )
        channel_name = f"orbiter:cancel:{task_id}"
        pubsub = r.pubsub()
        try:
            await pubsub.subscribe(channel_name)  # type: ignore[misc]
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if msg is not None and msg["type"] == "message":
                    token.cancel()
                    return
                await asyncio.sleep(0.01)
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.aclose()  # type: ignore[misc]
            await r.aclose()

    def _reconstruct_agent(self, agent_config: dict[str, Any]) -> Any:
        """Reconstruct an Agent or Swarm from the serialized config dict."""
        if "agents" in agent_config:
            # Swarm config
            from orbiter.swarm import Swarm  # pyright: ignore[reportMissingImports]

            return Swarm.from_dict(agent_config)
        else:
            from orbiter.agent import Agent  # pyright: ignore[reportMissingImports]

            return Agent.from_dict(agent_config)

    async def _run_agent(
        self, agent: Any, task: TaskPayload, token: CancellationToken
    ) -> str:
        """Stream agent execution, publishing events and collecting output.

        Checks *token* between steps for cooperative cancellation.  When
        cancelled, emits a ``StatusEvent(status='cancelled')`` and stops.
        """
        from orbiter.runner import run  # pyright: ignore[reportMissingImports]
        from orbiter.types import StatusEvent, TextEvent  # pyright: ignore[reportMissingImports]

        messages = _deserialize_messages(task.messages) if task.messages else None
        text_parts: list[str] = []

        async for event in run.stream(
            agent,
            task.input,
            messages=messages,
            detailed=task.detailed,
        ):
            # Check for cancellation between steps
            if token.cancelled:
                cancelled_event = StatusEvent(
                    status="cancelled",
                    agent_name=getattr(agent, "name", ""),
                    message=f"Task {task.task_id} cancelled",
                )
                await self._publisher.publish(task.task_id, cancelled_event)
                break

            # Publish every event to Redis
            await self._publisher.publish(task.task_id, event)

            # Collect text for final result
            if isinstance(event, TextEvent):
                text_parts.append(event.text)

        return "".join(text_parts)
