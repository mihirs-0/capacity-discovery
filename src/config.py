"""Dataclass-based configuration with defaults."""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class TaskConfig:
    K: int = 20
    n_b: int = 500
    len_b: int = 6
    len_a: int = 4
    len_z: int = 2
    vocab_size: int = 36
    data_seed: int = 42

    @property
    def D(self) -> int:
        return self.n_b * self.K


@dataclass
class ModelConfig:
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_mlp: int = 512
    vocab_size: int = 40
    max_seq_len: int = 16

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    eps: float = 1e-8
    batch_size: int = 128
    warmup_steps: int = 500
    max_steps: int = 50_000
    model_seed: int = 0


@dataclass
class EvalConfig:
    eval_every: int = 100
    checkpoint_every: int = 200
    z_shuffle_batch_size: int = 1024
    group_eval_n_groups: int = 200
    hessian_enabled: bool = False
    d7_enabled: bool = False


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "default"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add derived fields
        d["task"]["D"] = self.task.D
        d["model"]["d_head"] = self.model.d_head
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        task = TaskConfig(**{k: v for k, v in d.get("task", {}).items() if k != "D"})
        model = ModelConfig(**{k: v for k, v in d.get("model", {}).items() if k not in ("d_head", "param_count")})
        training = TrainingConfig(**d.get("training", {}))
        eval_cfg = EvalConfig(**d.get("eval", {}))
        name = d.get("experiment_name", "default")
        return cls(task=task, model=model, training=training, eval=eval_cfg, experiment_name=name)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
