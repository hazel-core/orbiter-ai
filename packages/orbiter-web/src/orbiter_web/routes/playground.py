"""Playground WebSocket endpoint for streaming chat with agents."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from orbiter_web.database import get_db

router = APIRouter(tags=["playground"])


# ---- Persistence helpers ----


async def _create_conversation(agent_id: str, user_id: str) -> str:
    """Create a new conversation and return its id."""
    conv_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO conversations (id, agent_id, user_id) VALUES (?, ?, ?)",
            (conv_id, agent_id, user_id),
        )
        await db.commit()
    return conv_id


async def _save_message(
    conversation_id: str,
    role: str,
    content: str,
    usage_json: str | None = None,
) -> str:
    """Save a message to the database and return its id."""
    msg_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, usage_json) VALUES (?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, usage_json),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()
    return msg_id


async def _load_conversation_messages(conversation_id: str) -> list[dict[str, str]]:
    """Load all messages from a conversation for the history."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


class _TokenCollector:
    """Wraps a WebSocket to collect streamed token content."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self.collected = ""
        self.usage: dict[str, int] | None = None

    async def send_json(self, data: Any) -> None:
        if data.get("type") == "token":
            self.collected += data.get("content", "")
        elif data.get("type") == "done":
            self.usage = data.get("usage")
        await self._ws.send_json(data)


async def _get_user_from_cookie(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on the WebSocket connection."""
    session_id = websocket.cookies.get("orbiter_session")
    if not session_id:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (session_id,),
        )
        row = await cursor.fetchone()

    return dict(row) if row else None


async def _get_agent(agent_id: str, user_id: str) -> dict[str, Any] | None:
    """Load agent config from DB."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agents WHERE id = ? AND user_id = ?",
            (agent_id, user_id),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def _resolve_api_key(provider_id: str, user_id: str) -> tuple[dict[str, Any] | None, str]:
    """Resolve provider config and API key. Returns (provider_dict, api_key)."""
    from orbiter_web.crypto import decrypt_api_key

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None, ""
        provider = dict(row)

        api_key = ""
        if provider.get("encrypted_api_key"):
            api_key = decrypt_api_key(provider["encrypted_api_key"])
        else:
            cursor = await db.execute(
                "SELECT encrypted_api_key FROM provider_keys WHERE provider_id = ? AND status = 'active' LIMIT 1",
                (provider_id,),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_api_key"])

    return provider, api_key


async def _find_provider_by_type(provider_type: str, user_id: str) -> str | None:
    """Find a provider ID by type for the given user."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id FROM providers
            WHERE provider_type = ? AND user_id = ?
            AND (
                encrypted_api_key IS NOT NULL AND encrypted_api_key != ''
                OR EXISTS (
                    SELECT 1 FROM provider_keys pk
                    WHERE pk.provider_id = providers.id AND pk.status = 'active'
                )
            )
            LIMIT 1
            """,
            (provider_type, user_id),
        )
        row = await cursor.fetchone()
    return row["id"] if row else None


async def _stream_openai(
    websocket: WebSocket,
    api_key: str,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> None:
    """Stream a response from an OpenAI-compatible API."""
    import httpx

    url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    prompt_tokens = 0
    completion_tokens = 0

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        ) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Extract usage from final chunk if present
            if usage := chunk.get("usage"):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    await websocket.send_json({"type": "token", "content": content})

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


async def _stream_anthropic(
    websocket: WebSocket,
    api_key: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> None:
    """Stream a response from the Anthropic API."""
    import httpx

    url = "https://api.anthropic.com/v1/messages"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": [m for m in messages if m["role"] != "system"],
        "stream": True,
        "max_tokens": max_tokens or 1024,
        "temperature": temperature,
    }
    if system_prompt:
        body["system"] = system_prompt

    input_tokens = 0
    output_tokens = 0

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream(
            "POST",
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
        ) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        await websocket.send_json({"type": "token", "content": text})
            elif event_type == "message_start":
                usage = event.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
            elif event_type == "message_delta":
                usage = event.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        }
    )


async def _stream_gemini(
    websocket: WebSocket,
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> None:
    """Stream a response from the Gemini API."""
    import httpx

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"

    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if msg["role"] == "system":
            role = "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    body: dict[str, Any] = {"contents": contents}
    if temperature or max_tokens:
        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        body["generationConfig"] = gen_config

    prompt_tokens = 0
    completion_tokens = 0

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream("POST", url, json=body) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            candidates = chunk.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    text = part.get("text", "")
                    if text:
                        await websocket.send_json({"type": "token", "content": text})

            usage_meta = chunk.get("usageMetadata", {})
            if usage_meta:
                prompt_tokens = usage_meta.get("promptTokenCount", prompt_tokens)
                completion_tokens = usage_meta.get("candidatesTokenCount", completion_tokens)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


async def _stream_ollama(
    websocket: WebSocket,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> None:
    """Stream a response from an Ollama API."""
    import httpx

    url = (base_url or "http://localhost:11434") + "/api/chat"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        body["options"] = {"temperature": temperature}

    prompt_tokens = 0
    completion_tokens = 0

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream("POST", url, json=body) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = chunk.get("message", {}).get("content", "")
            if content:
                await websocket.send_json({"type": "token", "content": content})

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


@router.websocket("/api/playground/{agent_id}/chat")
async def playground_chat(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket endpoint for streaming chat with an agent."""
    # Authenticate via session cookie
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    # Load agent
    agent = await _get_agent(agent_id, user["id"])
    if agent is None:
        await websocket.close(code=4004, reason="Agent not found")
        return

    await websocket.accept()

    # Resolve provider and API key
    provider_type = agent.get("model_provider", "")
    model_name = agent.get("model_name", "")
    if not provider_type or not model_name:
        await websocket.send_json({"type": "error", "message": "Agent has no model configured"})
        await websocket.close()
        return

    provider_id = await _find_provider_by_type(provider_type, user["id"])
    if not provider_id:
        await websocket.send_json(
            {"type": "error", "message": f"No {provider_type} provider configured"}
        )
        await websocket.close()
        return

    provider, api_key = await _resolve_api_key(provider_id, user["id"])
    if not api_key:
        await websocket.send_json(
            {"type": "error", "message": "No API key configured for provider"}
        )
        await websocket.close()
        return

    base_url = (provider or {}).get("base_url", "") or ""
    temperature = agent.get("temperature") or 0.7
    max_tokens = agent.get("max_tokens")

    # Build system prompt from agent config
    system_prompt = ""
    persona_parts = []
    if agent.get("persona_role"):
        persona_parts.append(f"Role: {agent['persona_role']}")
    if agent.get("persona_goal"):
        persona_parts.append(f"Goal: {agent['persona_goal']}")
    if agent.get("persona_backstory"):
        persona_parts.append(f"Backstory: {agent['persona_backstory']}")
    if persona_parts:
        system_prompt = "## Persona\n" + "\n".join(persona_parts) + "\n\n"
    if agent.get("instructions"):
        system_prompt += agent["instructions"]

    # Conversation persistence
    conversation_id: str | None = None

    # Conversation history (backed by DB when conversation_id is set)
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            # Handle conversation loading
            if msg.get("type") == "load_conversation":
                conv_id = msg.get("conversation_id", "")
                if conv_id:
                    saved_messages = await _load_conversation_messages(conv_id)
                    if saved_messages:
                        conversation_id = conv_id
                        # Rebuild history: system prompt + saved messages
                        history = []
                        if system_prompt:
                            history.append({"role": "system", "content": system_prompt})
                        history.extend(saved_messages)
                        await websocket.send_json(
                            {
                                "type": "conversation_loaded",
                                "conversation_id": conv_id,
                                "messages": saved_messages,
                            }
                        )
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "Conversation not found"}
                        )
                continue

            content = msg.get("content", "").strip()
            if not content:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Create conversation on first real message if not already loaded
            if conversation_id is None:
                conversation_id = await _create_conversation(agent_id, user["id"])
                await websocket.send_json(
                    {"type": "conversation_created", "conversation_id": conversation_id}
                )
                # Save system prompt as first message
                if system_prompt:
                    await _save_message(conversation_id, "system", system_prompt)

            # Add user message to history and persist
            history.append({"role": "user", "content": content})
            await _save_message(conversation_id, "user", content)

            # Stream response, collecting tokens
            collector = _TokenCollector(websocket)
            try:
                if provider_type in ("openai", "custom"):
                    await _stream_openai(
                        collector, api_key, base_url, model_name, history, temperature, max_tokens
                    )  # type: ignore[arg-type]
                elif provider_type == "anthropic":
                    await _stream_anthropic(
                        collector,
                        api_key,
                        model_name,
                        system_prompt,
                        history,
                        temperature,
                        max_tokens,
                    )  # type: ignore[arg-type]
                elif provider_type == "gemini":
                    await _stream_gemini(
                        collector, api_key, model_name, history, temperature, max_tokens
                    )  # type: ignore[arg-type]
                elif provider_type == "ollama":
                    await _stream_ollama(collector, base_url, model_name, history, temperature)  # type: ignore[arg-type]
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unsupported provider: {provider_type}"}
                    )
                    continue
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"Stream error: {exc!s}"})
                continue

            # Persist the full assistant response
            assistant_content = collector.collected or "(empty response)"
            history.append({"role": "assistant", "content": assistant_content})
            usage_str = json.dumps(collector.usage) if collector.usage else None
            await _save_message(conversation_id, "assistant", assistant_content, usage_str)

    except WebSocketDisconnect:
        pass
