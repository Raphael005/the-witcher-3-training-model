from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_id: str
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    dataset_id: str
    eval_split_size: float = 0.1


@dataclass
class OutputConfig:
    output_root: str = "runs"
    run_name: str = "sft-run"

    @property
    def output_dir(self) -> Path:
        # SFT_OUTPUT_ROOT lets ops override the location without editing configs.
        root = Path(os.environ.get("SFT_OUTPUT_ROOT", self.output_root))
        return root / self.run_name


@dataclass
class TrainingConfig:
    num_train_epochs: float = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    auto_find_batch_size: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_bnb_8bit"
    max_length: int = 1024
    assistant_only_loss: bool = True
    loss_type: str = "nll"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    logging_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        return cls(
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw["data"]),
            output=OutputConfig(**raw["output"]),
            training=TrainingConfig(**raw.get("training", {})),
            seed=int(raw.get("seed", 42)),
        )
