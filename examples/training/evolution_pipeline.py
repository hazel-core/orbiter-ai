"""Evolution pipeline — iterative agent improvement via synthesis + training.

Demonstrates a full training lifecycle:
1. Create trajectory data from simulated agent runs
2. Synthesise additional training data via template augmentation
3. Run a multi-epoch evolution pipeline with a mock strategy
4. Inspect per-epoch metrics and final results

Usage:
    uv run python examples/training/evolution_pipeline.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from orbiter.train.evolution import (  # pyright: ignore[reportMissingImports]
    EvolutionConfig,
    EvolutionPipeline,
    EvolutionStrategy,
)
from orbiter.train.synthesis import (  # pyright: ignore[reportMissingImports]
    SynthesisConfig,
    SynthesisPipeline,
    filter_by_score,
)
from orbiter.train.trajectory import (  # pyright: ignore[reportMissingImports]
    TrajectoryDataset,
    TrajectoryItem,
)

# --- Generate seed trajectory data -------------------------------------------

SEED_DATA = [
    TrajectoryItem(task_id="t1", agent_id="bot", step=i, input=f"q{i}", output=f"a{i}", score=s)
    for i, s in enumerate([0.9, 0.6, 0.8, 0.4, 0.95, 0.7])
]


# --- Data synthesis ----------------------------------------------------------


def run_synthesis(items: Sequence[TrajectoryItem]) -> list[TrajectoryItem]:
    """Filter high-quality items and augment via template synthesis."""
    quality = filter_by_score(list(items), min_score=0.7)
    print(f"Filtered: {len(items)} → {len(quality)} items (score >= 0.7)")

    config = SynthesisConfig(num_samples=len(quality), strategy="template", seed=42)
    pipeline = SynthesisPipeline(config=config)
    result = asyncio.run(pipeline.run(quality))
    print(
        f"Synthesised: {result.total} items ({len(result.train_items)} train, {len(result.test_items)} test)"
    )
    return list(result.items)


# --- Custom evolution strategy (mock) ----------------------------------------


class DemoStrategy(EvolutionStrategy):
    """Mock strategy that simulates improvement over epochs."""

    async def synthesise(self, agent: Any, data: Sequence[Any], epoch: int) -> Sequence[Any]:
        return data

    async def train(self, agent: Any, data: Sequence[Any], epoch: int) -> dict[str, float]:
        loss = max(0.1, 1.0 - epoch * 0.25)
        return {"loss": loss, "steps": len(data)}

    async def evaluate(self, agent: Any, data: Sequence[Any], epoch: int) -> dict[str, float]:
        accuracy = min(0.99, 0.5 + epoch * 0.15)
        return {"accuracy": accuracy}


# --- Evolution loop ----------------------------------------------------------


async def run_evolution(data: list[TrajectoryItem]) -> None:
    """Run a multi-epoch evolution pipeline."""
    config = EvolutionConfig(max_epochs=4, early_stop_threshold=0.95)
    strategy = DemoStrategy()
    pipeline = EvolutionPipeline(config=config, strategy=strategy)

    result = await pipeline.run(agent=None, data=data)

    print(f"\nEvolution complete: {result.total_epochs} epochs")
    print(f"  Final accuracy: {result.final_accuracy:.2f}")
    print(f"  Early stopped:  {result.early_stopped}")
    print(f"  Best epoch:     {result.best_epoch}")
    for epoch in result.epochs:
        print(
            f"  Epoch {epoch.epoch}: loss={epoch.train_loss:.3f}, accuracy={epoch.eval_accuracy:.3f}"
        )


# --- Main --------------------------------------------------------------------


def main() -> None:
    print("=== Trajectory Seed Data ===")
    dataset = TrajectoryDataset()
    for item in SEED_DATA:
        dataset.append_trajectory(item)
    print(f"Seed dataset: {len(dataset)} items\n")

    print("=== Data Synthesis ===")
    augmented = run_synthesis(SEED_DATA)
    print()

    print("=== Evolution Pipeline ===")
    asyncio.run(run_evolution(augmented))


if __name__ == "__main__":
    main()
