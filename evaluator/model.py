from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LOGGER_NAME = "ft_linear_regression.evaluate"
EPSILON = 1e-12
OUTLIER_Z_THRESHOLD = 3.0


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float | None
    r2_defined: bool
    mean_error: float
    error_std: float
    max_abs_error: float
    outlier_count: int


@dataclass(frozen=True)
class MetricsComparison:
    model: RegressionMetrics
    baseline: RegressionMetrics
    delta_mse: float
    signal_to_noise_ratio: float | None
    usefulness_score: float | None


@dataclass(frozen=True)
class SplitInfo:
    test_ratio: float
    seed: int
    train_samples: int
    test_samples: int


@dataclass(frozen=True)
class EvaluationResult:
    samples: int
    full: MetricsComparison
    train: MetricsComparison
    test: MetricsComparison
    split: SplitInfo
    mileage_price_correlation: float | None


@dataclass(frozen=True)
class EvaluateArgs:
    dataset_path: Path
    model_path: Path
    json_output: bool
    report_path: Path | None
    test_ratio: float
    seed: int


@dataclass(frozen=True)
class ErrorStats:
    residuals: tuple[float, ...]
    abs_error_sum: float
    sq_error_sum: float
    max_abs_error: float
