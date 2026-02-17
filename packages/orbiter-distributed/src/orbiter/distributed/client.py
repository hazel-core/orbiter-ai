"""High-level API for submitting agent execution to the distributed queue."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from orbiter.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from orbiter.distributed.events import EventSubscriber  # pyright: ignore[reportMissingImports]
from orbiter.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)
from orbiter.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
from orbiter.types import StreamEvent  # pyright: ignore[reportMissingImports]

_DEFAULT_REDIS_ENV = "ORBITER_REDIS_URL"


class TaskHandle:
    """Handle returned by :func:`distributed` for monitoring a submitted task.

    Provides methods to wait for the result, stream live events, check
    status, and cancel the task.

    Args:
        task_id: The unique task identifier.
        broker: Connected :class:`TaskBroker` instance.
        store: Connected :class:`TaskStore` instance.
        subscriber: Connected :class:`EventSubscriber` instance.
    """

    def __init__(
        self,
        task_id: str,
        *,
        broker: TaskBroker,
        store: TaskStore,
        subscriber: EventSubscriber,
    ) -> None:
        self._task_id = task_id
        self._broker = broker
        self._store = store
        self._subscriber = subscriber

    @property
    def task_id(self) -> str:
        """The unique task identifier."""
        return self._task_id

    async def result(self, *, poll_interval: float = 0.5) -> dict[str, Any]:
        """Block until the task completes and return the result dict.

        Polls :class:`TaskStore` at *poll_interval* seconds until the task
        reaches a terminal status (``COMPLETED``, ``FAILED``, or
        ``CANCELLED``).

        Returns:
            The ``result`` dict from :class:`TaskResult` on success.

        Raises:
            RuntimeError: If the task failed or was cancelled.
        """
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        while True:
            task_result = await self._store.get_status(self._task_id)
            if task_result is not None and task_result.status in terminal:
                if task_result.status == TaskStatus.COMPLETED:
                    return task_result.result or {}
                if task_result.status == TaskStatus.CANCELLED:
                    msg = f"Task {self._task_id} was cancelled"
                    raise RuntimeError(msg)
                # FAILED
                msg = f"Task {self._task_id} failed: {task_result.error or 'unknown error'}"
                raise RuntimeError(msg)
            await asyncio.sleep(poll_interval)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Subscribe to live streaming events for this task.

        Yields deserialized :class:`StreamEvent` instances via Redis Pub/Sub.
        The iterator ends when a terminal ``StatusEvent`` is received.
        """
        async for event in self._subscriber.subscribe(self._task_id):
            yield event

    async def cancel(self) -> None:
        """Cancel the running task."""
        await self._broker.cancel(self._task_id)

    async def status(self) -> TaskResult | None:
        """Return the current task status."""
        return await self._store.get_status(self._task_id)


async def distributed(
    agent: Any,
    input: str,
    *,
    messages: list[dict[str, Any]] | None = None,
    redis_url: str | None = None,
    detailed: bool = False,
    timeout: float = 300.0,
    metadata: dict[str, Any] | None = None,
) -> TaskHandle:
    """Submit agent execution to the distributed queue.

    Serializes the agent (or swarm) to a :class:`TaskPayload`, enqueues it
    via :class:`TaskBroker`, and returns a :class:`TaskHandle` for
    monitoring the result.

    Args:
        agent: An ``Agent`` or ``Swarm`` instance (must support ``to_dict()``).
        input: The input string for the agent.
        messages: Optional message history.
        redis_url: Redis connection URL.  Defaults to the
            ``ORBITER_REDIS_URL`` environment variable.
        detailed: Whether to enable rich streaming events.
        timeout: Task timeout in seconds.
        metadata: Arbitrary metadata dict attached to the task payload.

    Returns:
        A :class:`TaskHandle` for result retrieval, streaming, and cancellation.

    Raises:
        ValueError: If *redis_url* is not provided and ``ORBITER_REDIS_URL``
            is not set.
    """
    url = redis_url or os.environ.get(_DEFAULT_REDIS_ENV)
    if url is None:
        msg = (
            "redis_url must be provided or ORBITER_REDIS_URL "
            "environment variable must be set"
        )
        raise ValueError(msg)

    broker = TaskBroker(url)
    store = TaskStore(url)
    subscriber = EventSubscriber(url)

    await broker.connect()
    await store.connect()
    await subscriber.connect()

    payload = TaskPayload(
        agent_config=agent.to_dict(),
        input=input,
        messages=messages or [],
        detailed=detailed,
        metadata=metadata or {},
        created_at=time.time(),
        timeout_seconds=timeout,
    )

    await broker.submit(payload)

    return TaskHandle(
        payload.task_id,
        broker=broker,
        store=store,
        subscriber=subscriber,
    )
