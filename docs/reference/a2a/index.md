# orbiter.a2a

Agent-to-Agent protocol for cross-service agent communication.

## Installation

```bash
pip install orbiter-a2a
```

## Module path

```python
import orbiter.a2a
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `A2AClient` | `orbiter.a2a.client` | HTTP client for communicating with remote A2A agents |
| `A2AClientError` | `orbiter.a2a.client` | Error raised for client-level errors |
| `ClientManager` | `orbiter.a2a.client` | Thread-safe manager for per-thread A2A clients |
| `RemoteAgent` | `orbiter.a2a.client` | Agent-compatible wrapper for remote A2A agents |
| `A2AServer` | `orbiter.a2a.server` | FastAPI-based A2A server with agent card discovery |
| `A2AServerError` | `orbiter.a2a.server` | Error raised for server-level errors |
| `AgentExecutor` | `orbiter.a2a.server` | Wraps an agent for A2A task execution |
| `TaskStore` | `orbiter.a2a.server` | Protocol for task state storage |
| `InMemoryTaskStore` | `orbiter.a2a.server` | In-memory task store for development |
| `AgentCard` | `orbiter.a2a.types` | Complete metadata descriptor for a remote agent |
| `AgentSkill` | `orbiter.a2a.types` | A single capability advertised by an agent |
| `AgentCapabilities` | `orbiter.a2a.types` | Runtime capabilities of an A2A agent |
| `TaskState` | `orbiter.a2a.types` | Lifecycle states for a remote A2A task |
| `TaskStatus` | `orbiter.a2a.types` | Current status of a remote A2A task |
| `TaskStatusUpdateEvent` | `orbiter.a2a.types` | Emitted when a remote task changes state |
| `TaskArtifactUpdateEvent` | `orbiter.a2a.types` | Emitted when a remote task produces output |
| `TransportMode` | `orbiter.a2a.types` | Supported A2A transport protocols |
| `ServingConfig` | `orbiter.a2a.types` | Server-side configuration for A2A publishing |
| `ClientConfig` | `orbiter.a2a.types` | Client-side configuration for connecting |

## Submodules

- [orbiter.a2a.types](types.md) -- Protocol types, agent cards, configs, and task events
- [orbiter.a2a.client](client.md) -- A2AClient, ClientManager, RemoteAgent
- [orbiter.a2a.server](server.md) -- A2AServer, AgentExecutor, TaskStore
