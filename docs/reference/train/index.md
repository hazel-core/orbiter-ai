# orbiter.train

Training framework with trajectory capture, data synthesis, evolution pipelines, and VeRL integration.

## Installation

```bash
pip install orbiter-train

# For VeRL integration:
pip install orbiter-train[verl]
```

## Module path

```python
import orbiter.train
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `TrajectoryItem` | `orbiter.train.trajectory` | A single step in an agent execution trajectory |
| `TrajectoryDataset` | `orbiter.train.trajectory` | Dataset of trajectory items with capture and export |
| `TrajectoryStrategy` | `orbiter.train.trajectory` | ABC for building trajectory items from messages |
| `DefaultStrategy` | `orbiter.train.trajectory` | Default strategy extracting input/output/tool_calls |
| `TrajectoryError` | `orbiter.train.trajectory` | Error during trajectory operations |
| `Trainer` | `orbiter.train.trainer` | Abstract base class for agent training |
| `TrainConfig` | `orbiter.train.trainer` | Minimal training configuration |
| `TrainerState` | `orbiter.train.trainer` | Trainer lifecycle state enum |
| `TrainMetrics` | `orbiter.train.trainer` | Metrics produced by training or evaluation |
| `TrainerError` | `orbiter.train.trainer` | Error during training operations |
| `DataSynthesiser` | `orbiter.train.synthesis` | ABC for custom data synthesisers |
| `TemplateSynthesiser` | `orbiter.train.synthesis` | Generate items via template transforms |
| `SynthesisPipeline` | `orbiter.train.synthesis` | Orchestrates data synthesis from trajectory items |
| `SynthesisConfig` | `orbiter.train.synthesis` | Configuration for a synthesis pipeline run |
| `SynthesisResult` | `orbiter.train.synthesis` | Output of a synthesis pipeline run |
| `SynthesisStrategy` | `orbiter.train.synthesis` | Strategy enum (llm, template, augment) |
| `SynthesisError` | `orbiter.train.synthesis` | Error during data synthesis |
| `augment_swap_io` | `orbiter.train.synthesis` | Augment by swapping input/output |
| `augment_add_noise` | `orbiter.train.synthesis` | Augment by adding noise to input |
| `filter_by_score` | `orbiter.train.synthesis` | Filter items by minimum score |
| `deduplicate` | `orbiter.train.synthesis` | Remove duplicate items |
| `split_dataset` | `orbiter.train.synthesis` | Split items into train/test sets |
| `EvolutionPipeline` | `orbiter.train.evolution` | Multi-epoch evolution pipeline |
| `EvolutionConfig` | `orbiter.train.evolution` | Configuration for an evolution run |
| `EvolutionStrategy` | `orbiter.train.evolution` | ABC for evolution phase strategies |
| `EvolutionState` | `orbiter.train.evolution` | Pipeline state enum |
| `EvolutionPhase` | `orbiter.train.evolution` | Evolution epoch phases enum |
| `EpochResult` | `orbiter.train.evolution` | Metrics for a single evolution epoch |
| `EvolutionResult` | `orbiter.train.evolution` | Aggregate result of a full evolution run |
| `EvolutionError` | `orbiter.train.evolution` | Error during evolution operations |
| `VeRLTrainer` | `orbiter.train.verl` | Concrete trainer integrating with VeRL |
| `VeRLConfig` | `orbiter.train.verl` | VeRL-specific training configuration |
| `VeRLAlgorithm` | `orbiter.train.verl` | Supported VeRL RL algorithms enum |
| `RewardSpec` | `orbiter.train.verl` | Descriptor for a reward function |

## Submodules

- [orbiter.train.trajectory](trajectory.md) -- TrajectoryDataset, TrajectoryItem, strategies
- [orbiter.train.trainer](trainer.md) -- Trainer ABC, TrainConfig, TrainMetrics
- [orbiter.train.synthesis](synthesis.md) -- DataSynthesiser, SynthesisPipeline, augment helpers
- [orbiter.train.evolution](evolution.md) -- EvolutionPipeline, EvolutionConfig, strategies
- [orbiter.train.verl](verl.md) -- VeRLTrainer, VeRLConfig, RewardSpec
