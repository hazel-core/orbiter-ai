"""Distributed worker process — claims tasks from the queue and executes agents."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import signal
import socket
import time
from typing import Any

from orbiter.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from orbiter.distributed.events import EventPublisher  # pyright: ignore[reportMissingImports]
from orbiter.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from orbiter.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]


def _generate_worker_id() -> str:
    """Generate a unique worker ID from hostname, PID, and a random suffix."""
    hostname = socket.gethostname()
    pid = os.getpid()
    suffix = random.randbytes(4).hex()
    return f"{hostname}-{pid}-{suffix}"


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
    """

    def __init__(
        self,
        redis_url: str,
        *,
        worker_id: str | None = None,
        concurrency: int = 1,
        queue_name: str = "orbiter:tasks",
        heartbeat_ttl: int = 30,
    ) -> None:
        self._redis_url = redis_url
        self._worker_id = worker_id or _generate_worker_id()
        self._concurrency = concurrency
        self._queue_name = queue_name
        self._heartbeat_ttl = heartbeat_ttl

        self._broker = TaskBroker(redis_url, queue_name=queue_name)
        self._store = TaskStore(redis_url)
        self._publisher = EventPublisher(redis_url)

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
        """
        self._started_at = time.time()

        await self._broker.connect()
        await self._store.connect()
        await self._publisher.connect()

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
        import redis.asyncio as aioredis

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

        try:
            # Mark as RUNNING
            await self._store.set_status(
                task.task_id,
                TaskStatus.RUNNING,
                worker_id=self._worker_id,
                started_at=time.time(),
            )

            # Reconstruct agent from config
            agent = self._reconstruct_agent(task.agent_config)

            # Execute via run.stream()
            result_text = await self._run_agent(agent, task)

            # Mark as COMPLETED
            await self._store.set_status(
                task.task_id,
                TaskStatus.COMPLETED,
                completed_at=time.time(),
                result={"output": result_text},
            )
            await self._broker.ack(task.task_id)
            self._tasks_processed += 1

        except Exception as exc:
            self._tasks_failed += 1
            await self._store.set_status(
                task.task_id,
                TaskStatus.FAILED,
                completed_at=time.time(),
                error=str(exc),
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

    def _reconstruct_agent(self, agent_config: dict[str, Any]) -> Any:
        """Reconstruct an Agent or Swarm from the serialized config dict."""
        if "agents" in agent_config:
            # Swarm config
            from orbiter.swarm import Swarm  # pyright: ignore[reportMissingImports]

            return Swarm.from_dict(agent_config)
        else:
            from orbiter.agent import Agent  # pyright: ignore[reportMissingImports]

            return Agent.from_dict(agent_config)

    async def _run_agent(self, agent: Any, task: TaskPayload) -> str:
        """Stream agent execution, publishing events and collecting output."""
        from orbiter.runner import run  # pyright: ignore[reportMissingImports]
        from orbiter.types import TextEvent  # pyright: ignore[reportMissingImports]

        text_parts: list[str] = []

        async for event in run.stream(
            agent,
            task.input,
            messages=None,
            detailed=task.detailed,
        ):
            # Publish every event to Redis
            await self._publisher.publish(task.task_id, event)

            # Collect text for final result
            if isinstance(event, TextEvent):
                text_parts.append(event.text)

        return "".join(text_parts)
