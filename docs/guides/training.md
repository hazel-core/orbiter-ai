# Training

The `orbiter-train` package provides a training framework with trajectory collection, data synthesis, evolutionary optimization, and VeRL (Verified Reinforcement Learning) integration. It supports the full training pipeline from collecting agent execution traces to fine-tuning with RLHF.

## Basic Usage

```python
from orbiter.train import TrajectoryDataset, TrajectoryItem, Trainer

# Collect trajectory data
dataset = TrajectoryDataset()
dataset.capture(TrajectoryItem(
    state="User asks about Python",
    action="search(query='Python programming')",
    reward=0.8,
))

# Export for training
dataset.export_json("training_data.json")
```

## Trajectory Collection

### TrajectoryItem

The fundamental data unit follows the **SAR (State-Action-Reward)** pattern:

```python
from orbiter.train import TrajectoryItem

item = TrajectoryItem(
    state="The user asked about Python decorators",
    action="respond(explanation='Decorators are...')",
    reward=0.9,
    metadata={"agent": "assistant", "step": 3},
)
```

### TrajectoryDataset

Collects and manages trajectory items:

```python
from orbiter.train import TrajectoryDataset

dataset = TrajectoryDataset()

# Capture items during agent execution
dataset.capture(TrajectoryItem(state="s1", action="a1", reward=0.7))
dataset.capture(TrajectoryItem(state="s2", action="a2", reward=0.9))

# Query the dataset
print(len(dataset))  # 2
items = dataset.items

# Export
dataset.export_json("trajectories.json")
dataset.export_csv("trajectories.csv")
```

### TrajectoryStrategy

Customize how trajectories are collected and processed:

```python
from orbiter.train import TrajectoryStrategy

class FilteredStrategy(TrajectoryStrategy):
    """Only keep high-reward trajectories."""

    def should_capture(self, item: TrajectoryItem) -> bool:
        return item.reward >= 0.5

    def process(self, item: TrajectoryItem) -> TrajectoryItem:
        # Normalize reward to [0, 1]
        return TrajectoryItem(
            state=item.state,
            action=item.action,
            reward=min(1.0, max(0.0, item.reward)),
        )

dataset = TrajectoryDataset(strategy=FilteredStrategy())
```

The `DefaultStrategy` accepts all items without modification.

## Data Synthesis

The synthesis module generates and augments training data:

### Pipeline-Based Synthesis

```python
from orbiter.train import SynthesisPipeline, SynthesisConfig, DataSynthesiser

config = SynthesisConfig(
    target_size=1000,
    min_score=0.7,
    dedup_threshold=0.9,
)

pipeline = SynthesisPipeline(config=config)
pipeline.add_step(my_synthesiser)
result = await pipeline.run(seed_data)

print(f"Generated: {result.generated_count}")
print(f"Filtered: {result.filtered_count}")
```

### Template-Based Synthesis

```python
from orbiter.train import TemplateSynthesiser

synthesiser = TemplateSynthesiser(
    templates=[
        "Explain {concept} in simple terms",
        "What are the pros and cons of {concept}?",
        "Compare {concept_a} with {concept_b}",
    ],
    variables={
        "concept": ["Python", "JavaScript", "Rust"],
        "concept_a": ["REST", "GraphQL"],
        "concept_b": ["gRPC", "WebSocket"],
    },
)

items = await synthesiser.generate(count=50)
```

### Data Augmentation Functions

Built-in augmentation utilities:

```python
from orbiter.train import (
    augment_add_noise,
    augment_swap_io,
    filter_by_score,
    deduplicate,
    split_dataset,
)

# Add noise to data (for robustness)
noisy_data = augment_add_noise(data, noise_level=0.1)

# Swap input/output for reverse-training
swapped = augment_swap_io(data)

# Filter low-quality items
filtered = filter_by_score(data, min_score=0.7)

# Remove near-duplicates
unique = deduplicate(data, threshold=0.9)

# Split into train/validation/test
train, val, test = split_dataset(data, ratios=[0.8, 0.1, 0.1])
```

## Trainer

The `Trainer` ABC defines a three-phase lifecycle: validation, training, and evaluation:

```python
from orbiter.train import Trainer, TrainConfig, TrainerState, TrainMetrics

class MyTrainer(Trainer):
    async def check_agent(self) -> bool:
        """Validate agent configuration."""
        return self.agent is not None

    async def check_dataset(self) -> bool:
        """Validate training dataset."""
        return len(self.dataset) > 0

    async def check_reward(self) -> bool:
        """Validate reward function."""
        return self.reward_fn is not None

    async def check_config(self) -> bool:
        """Validate training config."""
        return self.config.epochs > 0

    async def train(self) -> TrainMetrics:
        """Run training loop."""
        # ... training logic ...
        return TrainMetrics(loss=0.5, accuracy=0.85)

    async def evaluate(self) -> TrainMetrics:
        """Evaluate the trained model."""
        # ... evaluation logic ...
        return TrainMetrics(loss=0.3, accuracy=0.90)
```

### TrainConfig

```python
from orbiter.train import TrainConfig

config = TrainConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    warmup_steps=100,
    eval_every=500,
)
```

### TrainerState

Tracks training progress:

| State | Description |
|-------|-------------|
| `CREATED` | Initial state |
| `VALIDATING` | Running pre-training checks |
| `VALIDATED` | All checks passed |
| `TRAINING` | Training in progress |
| `TRAINED` | Training complete |
| `EVALUATING` | Running evaluation |
| `COMPLETE` | All phases done |
| `ERROR` | An error occurred |

## Evolution Pipeline

The evolution module implements multi-epoch iterative improvement:

```python
from orbiter.train import EvolutionPipeline, EvolutionConfig, EvolutionStrategy

config = EvolutionConfig(
    epochs=5,
    population_size=20,
    selection_ratio=0.5,
    mutation_rate=0.1,
)

class MyStrategy(EvolutionStrategy):
    async def evaluate(self, candidate) -> float:
        """Score a candidate."""
        return await score_agent(candidate)

    async def select(self, population, scores) -> list:
        """Select top candidates."""
        sorted_pop = sorted(zip(scores, population), reverse=True)
        cutoff = int(len(population) * 0.5)
        return [p for _, p in sorted_pop[:cutoff]]

    async def mutate(self, candidate) -> Any:
        """Create a variant."""
        return mutate_agent(candidate)

pipeline = EvolutionPipeline(config=config, strategy=MyStrategy())
result = await pipeline.run(initial_population)

print(f"Best score: {result.best_score}")
print(f"Epochs completed: {result.epochs_completed}")
```

### EvolutionPhase

| Phase | Description |
|-------|-------------|
| `INIT` | Pipeline created |
| `EVALUATING` | Scoring the population |
| `SELECTING` | Selecting top candidates |
| `MUTATING` | Creating variants |
| `COMPLETE` | All epochs done |

## VeRL Integration

The VeRL module provides RLHF training with PPO and GRPO algorithms:

```python
from orbiter.train import VeRLTrainer, VeRLConfig, RewardSpec, VeRLAlgorithm

config = VeRLConfig(
    algorithm=VeRLAlgorithm.PPO,
    epochs=3,
    batch_size=16,
    learning_rate=1e-5,
    kl_coef=0.1,
    clip_range=0.2,
)

reward = RewardSpec(
    name="helpfulness",
    weight=1.0,
    fn=my_reward_function,  # (output: str) -> float
)

trainer = VeRLTrainer(
    config=config,
    rewards=[reward],
)

# Train
metrics = await trainer.train(dataset)
print(f"Loss: {metrics.loss}, Reward: {metrics.reward}")
```

### Algorithms

| Algorithm | Value | Description |
|-----------|-------|-------------|
| `PPO` | `"ppo"` | Proximal Policy Optimization |
| `GRPO` | `"grpo"` | Group Relative Policy Optimization |

### RewardSpec

Define multiple reward signals with weights:

```python
rewards = [
    RewardSpec(name="accuracy", weight=0.6, fn=accuracy_reward),
    RewardSpec(name="brevity", weight=0.2, fn=brevity_reward),
    RewardSpec(name="safety", weight=0.2, fn=safety_reward),
]
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `TrajectoryItem` | `orbiter.train` | SAR data point: `state`, `action`, `reward` |
| `TrajectoryDataset` | `orbiter.train` | Collect and export trajectory data |
| `TrajectoryStrategy` | `orbiter.train` | ABC for trajectory filtering/processing |
| `DefaultStrategy` | `orbiter.train` | Accept-all default strategy |
| `TrainConfig` | `orbiter.train` | Training hyperparameters |
| `Trainer` | `orbiter.train` | ABC for training lifecycle |
| `TrainerState` | `orbiter.train` | Training state machine |
| `TrainMetrics` | `orbiter.train` | Training metric results |
| `SynthesisConfig` | `orbiter.train` | Data synthesis configuration |
| `SynthesisPipeline` | `orbiter.train` | Multi-step synthesis pipeline |
| `DataSynthesiser` | `orbiter.train` | ABC for data generators |
| `TemplateSynthesiser` | `orbiter.train` | Template-based data generation |
| `EvolutionConfig` | `orbiter.train` | Evolution hyperparameters |
| `EvolutionPipeline` | `orbiter.train` | Multi-epoch evolutionary optimization |
| `EvolutionStrategy` | `orbiter.train` | ABC for evaluate/select/mutate |
| `VeRLConfig` | `orbiter.train` | VeRL training configuration |
| `VeRLTrainer` | `orbiter.train` | RLHF trainer (PPO/GRPO) |
| `VeRLAlgorithm` | `orbiter.train` | Enum: `PPO`, `GRPO` |
| `RewardSpec` | `orbiter.train` | Named, weighted reward function |
| `augment_add_noise` | `orbiter.train` | Add noise augmentation |
| `augment_swap_io` | `orbiter.train` | Swap input/output augmentation |
| `filter_by_score` | `orbiter.train` | Filter by minimum score |
| `deduplicate` | `orbiter.train` | Remove near-duplicates |
| `split_dataset` | `orbiter.train` | Split into train/val/test |
