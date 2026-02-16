"""Cost and pricing REST API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/costs", tags=["costs"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelPricing(BaseModel):
    model_name: str
    provider_type: str
    pricing_input: float | None
    pricing_output: float | None


# ---------------------------------------------------------------------------
# GET /api/costs/pricing — return pricing for all models
# ---------------------------------------------------------------------------


@router.get("/pricing", response_model=list[ModelPricing])
async def list_pricing(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return pricing information for all models available to the user."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT m.model_name, p.provider_type, m.pricing_input, m.pricing_output
            FROM models m
            JOIN providers p ON m.provider_id = p.id
            WHERE m.user_id = ?
            ORDER BY p.provider_type, m.model_name
            """,
            (user["id"],),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]
