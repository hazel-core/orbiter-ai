"""Workflow execution engine — runs DAGs in topological order."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from orbiter_web.database import get_db

# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def topological_sort(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[list[str]]:
    """Return nodes grouped into execution layers (parallel within each layer).

    Uses Kahn's algorithm. Each returned list is a set of node IDs that can
    execute concurrently because all their dependencies are satisfied.

    Raises ``ValueError`` if the graph contains a cycle.
    """
    node_ids = {n["id"] for n in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    children: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in node_ids and tgt in node_ids:
            in_degree[tgt] += 1
            children[src].append(tgt)

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    layers: list[list[str]] = []
    visited = 0

    while queue:
        layer = list(queue)
        queue.clear()
        layers.append(layer)
        visited += len(layer)
        for nid in layer:
            for child in children[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    if visited != len(node_ids):
        raise ValueError("Workflow graph contains a cycle")

    return layers


# ---------------------------------------------------------------------------
# Node executors (stub — each returns simulated output)
# ---------------------------------------------------------------------------


async def _execute_node(node: dict[str, Any]) -> dict[str, Any]:
    """Execute a single workflow node and return its output.

    This is a stub implementation that simulates execution for each node type.
    Real implementations will call LLM APIs, run code, make HTTP requests, etc.

    Returns a dict with keys: node_id, node_type, label, result, and optionally
    logs (str) and token_usage (dict) for inspection support.
    """
    node_type = node.get("data", {}).get("nodeType", node.get("type", "unknown"))
    node_data = node.get("data", {})

    # Simulate a tiny delay so concurrent execution is observable.
    await asyncio.sleep(0.01)

    result: dict[str, Any] = {
        "node_id": node["id"],
        "node_type": node_type,
        "label": node_data.get("label", ""),
        "result": f"Executed {node_type} node",
    }

    # Capture type-specific execution details for inspection.
    if node_type == "llm":
        prompt = node_data.get("prompt", "")
        result["result"] = f"LLM response for: {prompt[:100]}"
        result["logs"] = f"prompt: {prompt}\nresponse: {result['result']}"
        result["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elif node_type == "code":
        code = node_data.get("code", "")
        result["result"] = "Code executed successfully"
        result["logs"] = f"stdout: \nstderr: \ncode: {code[:200]}"
    elif node_type == "api":
        url = node_data.get("url", "")
        result["logs"] = f"request: {node_data.get('method', 'GET')} {url}\nresponse: 200 OK"

    return result


# ---------------------------------------------------------------------------
# Single-node execution
# ---------------------------------------------------------------------------


async def execute_single_node(
    run_id: str,
    workflow_id: str,
    user_id: str,
    node: dict[str, Any],
    mock_input: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single node in isolation with mock input data.

    Creates a workflow_run record and a single workflow_run_logs entry,
    then returns the execution result.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Mark run as running.
    await _update_run_status(run_id, "running", started_at=now)

    # Merge mock_input into node data so the node executor sees it.
    node_with_input = {**node, "data": {**node.get("data", {}), **mock_input}}

    log_id = str(uuid.uuid4())
    input_snapshot = json.dumps(node_with_input.get("data", {}))

    async with get_db() as db:
        await db.execute(
            "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, input_json) VALUES (?, ?, ?, 'running', ?, ?)",
            (log_id, run_id, node["id"], now, input_snapshot),
        )
        await db.commit()

    try:
        output = await _execute_node(node_with_input)
        completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        logs_text = output.pop("logs", None)
        token_usage = output.pop("token_usage", None)
        token_usage_json = json.dumps(token_usage) if token_usage else None

        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_run_logs SET status = 'completed', output_json = ?, logs_text = ?, token_usage_json = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                (json.dumps(output), logs_text, token_usage_json, completed_at, run_id, node["id"]),
            )
            await db.commit()

        await _update_run_status(run_id, "completed", completed_at=completed_at)

        return {
            "run_id": run_id,
            "node_id": node["id"],
            "status": "completed",
            "output": output,
            "logs": logs_text,
            "token_usage": token_usage,
        }

    except Exception as exc:
        error_msg = str(exc)
        completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                (error_msg, completed_at, run_id, node["id"]),
            )
            await db.commit()

        await _update_run_status(run_id, "failed", completed_at=completed_at, error=error_msg)

        return {
            "run_id": run_id,
            "node_id": node["id"],
            "status": "failed",
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Run manager — tracks active runs for cancellation
# ---------------------------------------------------------------------------

_active_runs: dict[str, asyncio.Event] = {}


def _register_run(run_id: str) -> asyncio.Event:
    """Register a run and return its cancellation event."""
    cancel_event = asyncio.Event()
    _active_runs[run_id] = cancel_event
    return cancel_event


def _unregister_run(run_id: str) -> None:
    _active_runs.pop(run_id, None)


def cancel_run(run_id: str) -> bool:
    """Signal cancellation for a run. Returns True if the run was found."""
    event = _active_runs.get(run_id)
    if event is None:
        return False
    event.set()
    return True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


async def execute_workflow(
    run_id: str,
    workflow_id: str,
    user_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    event_callback: Any | None = None,
) -> str:
    """Execute a workflow DAG and persist results.

    ``event_callback`` is an optional async callable receiving event dicts
    for real-time streaming (e.g., over WebSocket).

    Returns the final status: 'completed', 'failed', or 'cancelled'.
    """

    cancel_event = _register_run(run_id)

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            with contextlib.suppress(Exception):
                await event_callback(event)

    try:
        layers = topological_sort(nodes, edges)
    except ValueError as exc:
        await _update_run_status(run_id, "failed", error=str(exc))
        await emit({"type": "execution_completed", "status": "failed", "error": str(exc)})
        _unregister_run(run_id)
        return "failed"

    node_map = {n["id"]: n for n in nodes}
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Mark run as running.
    await _update_run_status(run_id, "running", started_at=now)

    final_status = "completed"
    variables: dict[str, Any] = {}

    for layer in layers:
        # Check cancellation before each layer.
        if cancel_event.is_set():
            final_status = "cancelled"
            break

        # Execute all nodes in this layer concurrently.
        tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
        for nid in layer:
            node = node_map.get(nid)
            if node is None:
                continue

            # Create log entry with input snapshot for inspection.
            log_id = str(uuid.uuid4())
            started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            input_snapshot = json.dumps(node.get("data", {}))
            async with get_db() as db:
                await db.execute(
                    "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, input_json) VALUES (?, ?, ?, 'running', ?, ?)",
                    (log_id, run_id, nid, started, input_snapshot),
                )
                await db.commit()

            await emit({"type": "node_started", "node_id": nid})
            tasks[nid] = asyncio.create_task(_execute_node(node))

        # Gather results.
        for nid, task in tasks.items():
            try:
                if cancel_event.is_set():
                    task.cancel()
                    final_status = "cancelled"
                    continue

                output = await task
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                logs_text = output.pop("logs", None)
                token_usage = output.pop("token_usage", None)
                token_usage_json = json.dumps(token_usage) if token_usage else None

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'completed', output_json = ?, logs_text = ?, token_usage_json = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                        (json.dumps(output), logs_text, token_usage_json, completed_at, run_id, nid),
                    )
                    await db.commit()

                variables[nid] = output.get("result", "")
                await emit({"type": "node_completed", "node_id": nid, "output": output, "variables": variables})

            except asyncio.CancelledError:
                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'skipped' WHERE run_id = ? AND node_id = ?",
                        (run_id, nid),
                    )
                    await db.commit()
                final_status = "cancelled"

            except Exception as exc:
                error_msg = str(exc)
                completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

                async with get_db() as db:
                    await db.execute(
                        "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                        (error_msg, completed_at, run_id, nid),
                    )
                    await db.commit()

                await emit({"type": "node_failed", "node_id": nid, "error": error_msg})
                final_status = "failed"
                break  # Stop on first failure

        if final_status in ("failed", "cancelled"):
            break

    # Finalize.
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    error_msg = None if final_status == "completed" else (f"Workflow {final_status}")
    await _update_run_status(run_id, final_status, completed_at=completed_at, error=error_msg)

    # Update last_run_at on the workflow itself.
    async with get_db() as db:
        await db.execute(
            "UPDATE workflows SET last_run_at = ? WHERE id = ?",
            (completed_at, workflow_id),
        )
        await db.commit()

    await emit({"type": "execution_completed", "status": final_status, "variables": variables})
    _unregister_run(run_id)
    return final_status


# ---------------------------------------------------------------------------
# Debug execution — step-through mode
# ---------------------------------------------------------------------------

# Tracks active debug sessions: run_id -> command queue
_debug_sessions: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def register_debug_session(run_id: str) -> asyncio.Queue[dict[str, Any]]:
    """Register a debug session and return its command queue."""
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _debug_sessions[run_id] = q
    return q


def unregister_debug_session(run_id: str) -> None:
    _debug_sessions.pop(run_id, None)


def send_debug_command(run_id: str, command: dict[str, Any]) -> bool:
    """Send a command to a debug session. Returns True if session found."""
    q = _debug_sessions.get(run_id)
    if q is None:
        return False
    q.put_nowait(command)
    return True


async def execute_workflow_debug(
    run_id: str,
    workflow_id: str,
    user_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    command_queue: asyncio.Queue[dict[str, Any]],
    event_callback: Any | None = None,
) -> str:
    """Execute a workflow in debug mode — pauses before each node.

    The ``command_queue`` receives debug commands from the client:
    - ``{action: 'continue'}`` — execute the current node and advance
    - ``{action: 'skip'}`` — skip the current node and advance
    - ``{action: 'stop'}`` — abort execution
    - ``{action: 'set_breakpoint', node_id: str}`` — toggle breakpoint on a node
    - ``{action: 'set_variable', name: str, value: Any}`` — modify a variable

    Returns the final status: 'completed', 'failed', or 'cancelled'.
    """

    cancel_event = _register_run(run_id)
    breakpoints: set[str] = set()
    variables: dict[str, Any] = {}

    async def emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            with contextlib.suppress(Exception):
                await event_callback(event)

    try:
        layers = topological_sort(nodes, edges)
    except ValueError as exc:
        await _update_run_status(run_id, "failed", error=str(exc))
        await emit({"type": "execution_completed", "status": "failed", "error": str(exc)})
        _unregister_run(run_id)
        return "failed"

    node_map = {n["id"]: n for n in nodes}

    # Flatten layers into a sequential execution order for debug stepping.
    execution_order: list[str] = []
    for layer in layers:
        execution_order.extend(layer)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await _update_run_status(run_id, "running", started_at=now)

    final_status = "completed"

    for nid in execution_order:
        if cancel_event.is_set():
            final_status = "cancelled"
            break

        node = node_map.get(nid)
        if node is None:
            continue

        # Emit paused event — client should display controls.
        await emit({
            "type": "debug_paused",
            "node_id": nid,
            "variables": variables,
            "breakpoints": list(breakpoints),
        })

        # Wait for a command from the client.
        action = await _wait_for_debug_command(
            command_queue, cancel_event, breakpoints, variables
        )

        if action == "stop" or cancel_event.is_set():
            final_status = "cancelled"
            break

        if action == "skip":
            # Log the node as skipped.
            log_id = str(uuid.uuid4())
            started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
            async with get_db() as db:
                await db.execute(
                    "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, completed_at) VALUES (?, ?, ?, 'skipped', ?, ?)",
                    (log_id, run_id, nid, started, started),
                )
                await db.commit()
            await emit({"type": "node_skipped", "node_id": nid})
            continue

        # action == "continue" — execute the node.
        log_id = str(uuid.uuid4())
        started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        input_snapshot = json.dumps(node.get("data", {}))
        async with get_db() as db:
            await db.execute(
                "INSERT INTO workflow_run_logs (id, run_id, node_id, status, started_at, input_json) VALUES (?, ?, ?, 'running', ?, ?)",
                (log_id, run_id, nid, started, input_snapshot),
            )
            await db.commit()

        await emit({"type": "node_started", "node_id": nid})

        try:
            output = await _execute_node(node)
            completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            logs_text = output.pop("logs", None)
            token_usage = output.pop("token_usage", None)
            token_usage_json = json.dumps(token_usage) if token_usage else None

            async with get_db() as db:
                await db.execute(
                    "UPDATE workflow_run_logs SET status = 'completed', output_json = ?, logs_text = ?, token_usage_json = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                    (json.dumps(output), logs_text, token_usage_json, completed_at, run_id, nid),
                )
                await db.commit()

            # Store node output in variables for downstream inspection.
            variables[nid] = output.get("result", "")

            await emit({
                "type": "node_completed",
                "node_id": nid,
                "output": output,
                "variables": variables,
            })

        except Exception as exc:
            error_msg = str(exc)
            completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            async with get_db() as db:
                await db.execute(
                    "UPDATE workflow_run_logs SET status = 'failed', error = ?, completed_at = ? WHERE run_id = ? AND node_id = ?",
                    (error_msg, completed_at, run_id, nid),
                )
                await db.commit()

            await emit({"type": "node_failed", "node_id": nid, "error": error_msg})
            final_status = "failed"
            break

    # Finalize.
    completed_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    error_msg = None if final_status == "completed" else f"Workflow {final_status}"
    await _update_run_status(run_id, final_status, completed_at=completed_at, error=error_msg)

    async with get_db() as db:
        await db.execute(
            "UPDATE workflows SET last_run_at = ? WHERE id = ?",
            (completed_at, workflow_id),
        )
        await db.commit()

    await emit({"type": "execution_completed", "status": final_status, "variables": variables})
    _unregister_run(run_id)
    unregister_debug_session(run_id)
    return final_status


async def _wait_for_debug_command(
    command_queue: asyncio.Queue[dict[str, Any]],
    cancel_event: asyncio.Event,
    breakpoints: set[str],
    variables: dict[str, Any],
) -> str:
    """Wait for a continue/skip/stop command, processing side-effect commands inline.

    Side-effect commands (set_breakpoint, set_variable) are handled immediately
    and we keep waiting for a flow-control command.

    Returns: 'continue', 'skip', or 'stop'.
    """
    while True:
        if cancel_event.is_set():
            return "stop"

        try:
            cmd = await asyncio.wait_for(command_queue.get(), timeout=0.5)
        except TimeoutError:
            continue

        action = cmd.get("action", "")

        if action == "set_breakpoint":
            node_id = cmd.get("node_id", "")
            if node_id in breakpoints:
                breakpoints.discard(node_id)
            else:
                breakpoints.add(node_id)
            continue

        if action == "set_variable":
            name = cmd.get("name", "")
            value = cmd.get("value")
            if name:
                variables[name] = value
            continue

        if action in ("continue", "skip", "stop"):
            return action

        # Unknown action — ignore and keep waiting.


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _update_run_status(
    run_id: str,
    status: str,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
) -> None:
    """Update the status (and optional timestamps) of a workflow run."""
    fields = ["status = ?"]
    values: list[Any] = [status]

    if started_at:
        fields.append("started_at = ?")
        values.append(started_at)
    if completed_at:
        fields.append("completed_at = ?")
        values.append(completed_at)
    if error is not None:
        fields.append("error = ?")
        values.append(error)

    values.append(run_id)

    async with get_db() as db:
        await db.execute(
            f"UPDATE workflow_runs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        await db.commit()
