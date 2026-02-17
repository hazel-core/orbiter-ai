# orbiter-memory

Pluggable memory backends for short-term, long-term, and vector-based memory storage.

## Module Path

```
orbiter.memory
```

## Installation

```bash
pip install orbiter-memory
```

## Overview

The `orbiter-memory` package provides a memory system for agents with:

- **MemoryStore protocol** -- pluggable backend interface for add/get/search/clear
- **Typed memory items** -- `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory` with status lifecycle
- **ShortTermMemory** -- in-memory conversation store with scope filtering and windowing
- **LongTermMemory** -- persistent knowledge with deduplication and LLM-powered extraction
- **Summary system** -- configurable triggers and multi-template summarization
- **Event integration** -- `MemoryEventEmitter` wraps any store with EventBus events
- **Backend implementations** -- SQLite, Postgres, and Vector (embedding-based) stores

## Exports

| Export | Type | Description |
|---|---|---|
| `MemoryStore` | Protocol | Pluggable backend interface |
| `MemoryItem` | Pydantic model | Base memory entry |
| `MemoryMetadata` | Pydantic model | Scoping metadata |
| `MemoryStatus` | StrEnum | Item lifecycle status |
| `MemoryError` | Exception | Memory operation errors |
| `SystemMemory` | Pydantic model | System/initialization memory |
| `HumanMemory` | Pydantic model | User message memory |
| `AIMemory` | Pydantic model | AI response memory |
| `ToolMemory` | Pydantic model | Tool result memory |
| `ShortTermMemory` | Class | In-memory conversation store |
| `LongTermMemory` | Class | Persistent knowledge store |
| `Extractor` | Protocol | LLM extraction interface |
| `ExtractionType` | StrEnum | Knowledge extraction categories |
| `ExtractionTask` | Dataclass | Extraction task tracking |
| `TaskStatus` | StrEnum | Extraction task lifecycle |
| `MemoryOrchestrator` | Class | Async extraction orchestrator |
| `OrchestratorConfig` | Dataclass | Orchestrator settings |
| `Summarizer` | Protocol | LLM summarization interface |
| `SummaryConfig` | Dataclass | Summary trigger/generation settings |
| `SummaryResult` | Dataclass | Summary generation output |
| `SummaryTemplate` | StrEnum | Built-in summary templates |
| `check_trigger` | Function | Check if summary should be generated |
| `generate_summary` | Function | Generate summaries from items |
| `MemoryEventEmitter` | Class | EventBus-integrated store wrapper |
| `MEMORY_ADDED` | `str` | Event constant: `"memory:added"` |
| `MEMORY_SEARCHED` | `str` | Event constant: `"memory:searched"` |
| `MEMORY_CLEARED` | `str` | Event constant: `"memory:cleared"` |

## Import Patterns

```python
# Common imports
from orbiter.memory import (
    MemoryStore, MemoryItem, MemoryMetadata, MemoryStatus,
    ShortTermMemory, LongTermMemory,
    HumanMemory, AIMemory, SystemMemory, ToolMemory,
)

# Summary system
from orbiter.memory import (
    SummaryConfig, SummaryTemplate, SummaryResult,
    Summarizer, check_trigger, generate_summary,
)

# Backends (import directly from submodules)
from orbiter.memory.backends.sqlite import SQLiteMemoryStore
from orbiter.memory.backends.postgres import PostgresMemoryStore
from orbiter.memory.backends.vector import VectorMemoryStore, Embeddings, OpenAIEmbeddings
```

## Quick Example

```python
import asyncio
from orbiter.memory import (
    ShortTermMemory, HumanMemory, AIMemory, MemoryMetadata,
)

async def main():
    memory = ShortTermMemory(scope="session", max_rounds=10)

    meta = MemoryMetadata(user_id="user-1", session_id="sess-1")

    await memory.add(HumanMemory(content="Hello!", metadata=meta))
    await memory.add(AIMemory(content="Hi there!", metadata=meta))

    results = await memory.search(metadata=meta, limit=5)
    for item in results:
        print(f"[{item.memory_type}] {item.content}")

asyncio.run(main())
```

## See Also

- [base](base.md) -- MemoryStore protocol and typed items
- [short-term](short-term.md) -- ShortTermMemory
- [long-term](long-term.md) -- LongTermMemory and orchestrator
- [summary](summary.md) -- Summary system
- [events](events.md) -- MemoryEventEmitter
- [backends](backends.md) -- SQLite, Postgres, Vector stores
