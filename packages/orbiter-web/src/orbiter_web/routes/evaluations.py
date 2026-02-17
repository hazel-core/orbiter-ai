"""Evaluation framework REST API.

Provides CRUD for evaluation suites, running evaluations against agents,
and retrieving scored results.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.pagination import paginate
from orbiter_web.routes.auth import get_current_user
from orbiter_web.sanitize import sanitize_html
from orbiter_web.services.evaluators import EVALUATORS, run_evaluator

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    input: str = Field(..., min_length=1)
    expected: str = ""
    evaluator: str = "exact_match"


class EvaluationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    agent_id: str = Field(..., min_length=1)
    test_cases: list[TestCase] = Field(default_factory=list)


class EvaluationUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    agent_id: str | None = Field(None, min_length=1)
    test_cases: list[TestCase] | None = None


class EvaluationResponse(BaseModel):
    id: str
    name: str
    agent_id: str
    test_cases_json: str
    user_id: str
    created_at: str


class EvalResultResponse(BaseModel):
    id: str
    evaluation_id: str
    run_at: str
    results_json: str
    overall_score: float
    pass_rate: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


async def _send_to_agent(
    agent_id: str, user_id: str, message: str
) -> str:
    """Send a message to an agent and return the text response."""
    from orbiter_web.services.agent_runtime import _load_agent_row, _resolve_provider

    row = await _load_agent_row(agent_id)
    provider_type = row.get("model_provider", "")
    model_name = row.get("model_name", "")
    if not provider_type or not model_name:
        return "[error: agent has no model configured]"

    try:
        provider = await _resolve_provider(provider_type, model_name, user_id)
        instructions = row.get("instructions", "")
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": message})
        resp = await provider.complete(messages=messages, model=model_name)
        return resp.content
    except Exception as exc:
        return f"[error: {exc}]"


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_evaluations(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """List evaluation suites for the current user."""
    async with get_db() as db:
        result = await paginate(
            db,
            table="evaluations",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=EvaluationResponse, status_code=201)
async def create_evaluation(
    body: EvaluationCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new evaluation suite."""
    eval_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Validate agent exists
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?",
            (body.agent_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        test_cases_json = json.dumps(
            [tc.model_dump() for tc in body.test_cases]
        )

        await db.execute(
            """
            INSERT INTO evaluations (id, name, agent_id, test_cases_json, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                sanitize_html(body.name),
                body.agent_id,
                test_cases_json,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ?", (eval_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single evaluation suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _row_to_dict(row)


@router.put("/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
    evaluation_id: str,
    body: EvaluationUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an evaluation suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = sanitize_html(body.name)
        if body.agent_id is not None:
            # Validate agent
            cur = await db.execute(
                "SELECT id FROM agents WHERE id = ? AND user_id = ?",
                (body.agent_id, user["id"]),
            )
            if await cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            updates["agent_id"] = body.agent_id
        if body.test_cases is not None:
            updates["test_cases_json"] = json.dumps(
                [tc.model_dump() for tc in body.test_cases]
            )

        if not updates:
            raise HTTPException(status_code=422, detail="No fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), evaluation_id]
        await db.execute(
            f"UPDATE evaluations SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{evaluation_id}", status_code=204)
async def delete_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an evaluation suite and all its results."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        await db.execute(
            "DELETE FROM evaluations WHERE id = ?", (evaluation_id,)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


@router.post("/{evaluation_id}/run", response_model=EvalResultResponse)
async def run_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Run all test cases against the linked agent and score the responses."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = _row_to_dict(row)

    try:
        test_cases: list[dict[str, Any]] = json.loads(
            evaluation["test_cases_json"]
        )
    except (json.JSONDecodeError, TypeError):
        test_cases = []

    if not test_cases:
        raise HTTPException(
            status_code=422, detail="No test cases defined"
        )

    agent_id = evaluation["agent_id"]
    results: list[dict[str, Any]] = []
    total_score = 0.0
    pass_count = 0

    for tc in test_cases:
        input_msg = tc.get("input", "")
        expected = tc.get("expected", "")
        evaluator_type = tc.get("evaluator", "exact_match")

        # Validate evaluator type
        if evaluator_type not in EVALUATORS:
            evaluator_type = "exact_match"

        # Send to agent
        actual = await _send_to_agent(agent_id, user["id"], input_msg)

        # Score the response
        kwargs: dict[str, Any] = {}
        if evaluator_type == "llm_as_judge":
            from orbiter_web.services.agent_runtime import _resolve_provider

            async with get_db() as db2:
                cur = await db2.execute(
                    "SELECT model_provider, model_name FROM agents WHERE id = ?",
                    (agent_id,),
                )
                agent_row = await cur.fetchone()

            if agent_row:
                agent_data = dict(agent_row)
                kwargs["provider_resolver"] = _resolve_provider
                kwargs["provider_type"] = agent_data.get("model_provider", "")
                kwargs["model_name"] = agent_data.get("model_name", "")
                kwargs["user_id"] = user["id"]

        score = await run_evaluator(evaluator_type, expected, actual, **kwargs)
        passed = score >= 0.5
        if passed:
            pass_count += 1
        total_score += score

        results.append({
            "input": input_msg,
            "expected": expected,
            "actual": actual,
            "evaluator": evaluator_type,
            "score": round(score, 4),
            "passed": passed,
        })

    num_cases = len(test_cases)
    overall_score = round(total_score / num_cases, 4) if num_cases else 0.0
    pass_rate = round(pass_count / num_cases, 4) if num_cases else 0.0

    # Persist the result
    result_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO eval_results (id, evaluation_id, run_at, results_json, overall_score, pass_rate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                evaluation_id,
                now,
                json.dumps(results),
                overall_score,
                pass_rate,
            ),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM eval_results WHERE id = ?", (result_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@router.get("/{evaluation_id}/results", response_model=list[EvalResultResponse])
async def list_eval_results(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all result runs for an evaluation suite."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        cursor = await db.execute(
            "SELECT * FROM eval_results WHERE evaluation_id = ? ORDER BY run_at DESC",
            (evaluation_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
