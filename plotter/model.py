from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from predictor.model import Model
from evaluator.model import MetricsComparison

DEFAULT_OUTPUT_BASENAME = "regression_plot"


@dataclass(frozen=True)
class PlotArgs:
    dataset_path: Path
    model_path: Path
    output_path: Path
    image_format: str
    show: bool
    theme: str
    x_axis: str
    report_dir: Path | None
    dpi: int


@dataclass(frozen=True)
class LoadedPlotData:
    mileages: list[float]
    prices: list[float]
    model: Model
    predictions: list[float]


@dataclass(frozen=True)
class PlotAnalysis:
    comparison: MetricsComparison
    correlation: float | None
    samples: int
    mileage_min: float
    mileage_max: float
    price_min: float
    price_max: float
    residuals: list[float]
    outlier_flags: list[bool]
    mean_error_model_space: float
    error_std_model_space: float
    mean_residual_plot_space: float
    residual_std_plot_space: float
    outlier_count: int
    baseline_value: float


@dataclass(frozen=True)
class AxisTransform:
    x_values: list[float]
    x_label: str
    normalized: bool
    mean: float | None
    std: float | None
