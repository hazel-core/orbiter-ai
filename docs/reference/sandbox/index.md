# orbiter.sandbox

Isolated execution environments for safe agent operation.

## Installation

```bash
pip install orbiter-sandbox

# For Kubernetes support:
pip install orbiter-sandbox[kubernetes]
```

## Module path

```python
import orbiter.sandbox
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `Sandbox` | `orbiter.sandbox.base` | Abstract sandbox providing isolated execution |
| `LocalSandbox` | `orbiter.sandbox.base` | Sandbox that executes on the local machine |
| `SandboxStatus` | `orbiter.sandbox.base` | Lifecycle states for a sandbox |
| `SandboxError` | `orbiter.sandbox.base` | Error raised for sandbox-level errors |
| `SandboxBuilder` | `orbiter.sandbox.builder` | Fluent builder for constructing sandbox instances |
| `KubernetesSandbox` | `orbiter.sandbox.kubernetes` | Sandbox that manages a Kubernetes pod |
| `FilesystemTool` | `orbiter.sandbox.tools` | Sandboxed filesystem tool with directory restrictions |
| `TerminalTool` | `orbiter.sandbox.tools` | Sandboxed terminal tool with command filtering |

## Submodules

- [orbiter.sandbox.base](base.md) -- Sandbox ABC, LocalSandbox, SandboxStatus, SandboxError
- [orbiter.sandbox.builder](builder.md) -- SandboxBuilder with fluent API
- [orbiter.sandbox.kubernetes](kubernetes.md) -- KubernetesSandbox for remote execution
- [orbiter.sandbox.tools](tools.md) -- FilesystemTool and TerminalTool
