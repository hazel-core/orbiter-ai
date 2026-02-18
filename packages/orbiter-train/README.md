# orbiter-train

Training framework for the [Orbiter](../../README.md) multi-agent framework. Collect trajectories, synthesize training data, and integrate with RLHF pipelines.

## Installation

```bash
pip install orbiter-train

# With VeRL integration
pip install orbiter-train[verl]
```

Requires Python 3.11+, `orbiter-core`, and `orbiter-models`.

## What's Included

- **Trajectory collection** -- capture agent execution traces for training data.
- **Data synthesis** -- generate training examples from agent interactions.
- **Evolution** -- mutate and evolve training data for diversity.
- **VeRL integration** -- connect to VeRL for RLHF/GRPO training loops.
- **Dataset management** -- load, filter, and batch trajectory datasets.

## Quick Example

```python
from orbiter.train import TrajectoryCollector

collector = TrajectoryCollector()
result = await collector.run(agent, "Solve this problem step by step...")

# Export trajectories for training
dataset = collector.to_dataset()
dataset.save("trajectories.jsonl")
```

## Documentation

- [Training Guide](../../docs/guides/training.md)
- [API Reference](../../docs/reference/train/)
