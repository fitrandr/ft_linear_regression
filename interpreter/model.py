from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LOGGER_NAME = "ft_linear_regression.interpret"


@dataclass(frozen=True)
class InterpretArgs:
    report_path: Path
    output_path: Path
    print_output: bool


@dataclass(frozen=True)
class ScopeMetrics:
    name: str
    model_mae: float
    model_rmse: float
    model_mse: float
    model_r2: float | None
    model_r2_defined: bool
    model_mean_error: float
    model_outlier_count: int
    baseline_mse: float
    baseline_rmse: float
    delta_mse: float
    signal_to_noise_ratio: float | None
    usefulness_score: float | None


@dataclass(frozen=True)
class InterpretedReport:
    samples: int
    train_samples: int
    test_samples: int
    test_ratio: float
    seed: int
    correlation: float | None
    full: ScopeMetrics
    train: ScopeMetrics
    test: ScopeMetrics
    overfit_gap: float | None
