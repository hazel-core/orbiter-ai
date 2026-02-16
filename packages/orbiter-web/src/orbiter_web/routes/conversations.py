"""Conversation persistence routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationResponse(BaseModel):
    id: str
    agent_id: str
    user_id: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    tool_calls_json: str | None = None
    usage_json: str | None = None
    created_at: str


@router.get("")
async def list_conversations(
    agent_id: str | None = None,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[ConversationResponse]:
    """List conversations, optionally filtered by agent_id."""
    async with get_db() as db:
        if agent_id:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE user_id = ? AND agent_id = ? ORDER BY updated_at DESC",
                (user["id"], agent_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
    return [ConversationResponse(**dict(r)) for r in rows]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> ConversationResponse:
    """Get a single conversation."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(**dict(row))


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> list[MessageResponse]:
    """List messages in a conversation, ordered chronologically."""
    # Verify conversation belongs to user
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        cursor = await db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    return [MessageResponse(**dict(r)) for r in rows]


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Delete a conversation and all its messages."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user["id"]),
        )
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        await db.commit()
    return {"status": "deleted"}
