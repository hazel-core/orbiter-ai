"""Trajectory capture — collect agent execution traces for training.

Demonstrates the State-Action-Reward (SAR) trajectory pattern:
1. Simulate agent turns (messages in, response out)
2. Capture each step as a ``TrajectoryItem``
3. Export the dataset to JSON and CSV for downstream training

Usage:
    uv run python examples/training/trajectory_capture.py
"""

from __future__ import annotations

from orbiter.train.trajectory import (  # pyright: ignore[reportMissingImports]
    DefaultStrategy,
    TrajectoryDataset,
    TrajectoryItem,
)

# --- Simulate a multi-step agent conversation --------------------------------

TASK_ID = "task-weather-001"
AGENT_ID = "weather-bot"

CONVERSATION: list[dict[str, object]] = [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    },
    {"role": "tool", "content": "Sunny, 22°C in Tokyo"},
    {"role": "assistant", "content": "The weather in Tokyo is sunny at 22°C."},
]


def main() -> None:
    dataset = TrajectoryDataset()

    # --- Method 1: Capture from raw messages using the default strategy ------
    item = dataset.from_messages(
        CONVERSATION,
        task_id=TASK_ID,
        agent_id=AGENT_ID,
    )
    print(f"Captured step {item.step}: input={item.input!r}, output={item.output!r}")

    # --- Method 2: Build items manually for scored trajectories --------------
    scored_item = TrajectoryItem(
        task_id=TASK_ID,
        agent_id=AGENT_ID,
        step=1,
        input="Follow-up: what about Osaka?",
        output="Osaka is partly cloudy at 19°C.",
        score=0.95,
        status="success",
        metadata={"evaluator": "human"},
    )
    dataset.append_trajectory(scored_item)
    print(f"Added scored item: score={scored_item.score}")

    # --- Export ---------------------------------------------------------------
    json_data = dataset.to_json()
    print(f"\nJSON export ({len(json_data)} chars):")
    print(json_data[:200], "..." if len(json_data) > 200 else "")

    csv_data = dataset.to_csv()
    print(f"\nCSV export ({len(csv_data)} chars):")
    for line in csv_data.splitlines()[:3]:
        print(line[:120])

    # --- Validation via strategy ---------------------------------------------
    strategy = DefaultStrategy()
    items = list(dataset.get_task_trajectory(TASK_ID))
    is_valid = strategy.validate(items)
    print(f"\nTrajectory valid: {is_valid}  ({len(items)} items)")


if __name__ == "__main__":
    main()
