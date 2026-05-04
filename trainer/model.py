from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EPSILON = 1e-12
LOGGER_NAME = "ft_linear_regression.train"
MODEL_VERSION = 1

DatasetSplit = tuple[list[float], list[float], list[float], list[float]]


@dataclass(frozen=True)
class LinearModel:
    theta0: float
    theta1: float
    normalized_theta0: float
    normalized_theta1: float
    km_mean: float
    km_std: float

    def predict(self, mileage: float) -> float:
        return self.theta0 + self.theta1 * mileage


@dataclass(frozen=True)
class TrainingLogEntry:
    iteration: int
    mse: float


@dataclass(frozen=True)
class TrainingResult:
    model: LinearModel
    learning_rate: float
    iterations: int
    iterations_ran: int
    log_every: int
    samples: int
    stopped_early: bool
    best_train_mse: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    history: tuple[TrainingLogEntry, ...]


@dataclass(frozen=True)
class EvaluationMetrics:
    train_mse: float
    test_mse: float


@dataclass(frozen=True)
class TrainArgs:
    dataset_path: Path
    model_path: Path
    learning_rate: float
    iterations: int
    test_ratio: float
    seed: int
    log_every: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    verbosity: str


@dataclass(frozen=True)
class PipelineOutput:
    model_path: Path
    training_result: TrainingResult
    metrics: EvaluationMetrics
    split: DatasetSplit
