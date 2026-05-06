#!/usr/bin/env python3
"""Evaluate trained model quality on a CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from predictor.engine import load_model, predict
from predictor.model import Model, ModelPolicy
from trainer.data import split_dataset

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


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Empty dataset: cannot compute mean.")
    return sum(values) / len(values)


def _validate_finite_series(name: str, values: list[float]) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name} at index {index}.")


def _parse_finite_float(raw_value: str, field: str, line_no: int) -> float:
    value = raw_value.strip()
    if value == "":
        raise ValueError(f"Missing {field} value at line {line_no}.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {field} value at line {line_no}: {raw_value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value at line {line_no}: {raw_value!r}.")
    return parsed


def load_dataset(path: Path) -> tuple[list[float], list[float]]:
    mileages: list[float] = []
    prices: list[float] = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        raw_headers = reader.fieldnames or []
        header_map: dict[str, str] = {}
        for header in raw_headers:
            if header is not None:
                header_map[header.strip()] = header

        if "km" not in header_map or "price" not in header_map:
            raise ValueError("CSV must contain 'km' and 'price' columns.")

        km_key = header_map["km"]
        price_key = header_map["price"]

        for line_no, row in enumerate(reader, start=2):
            raw_km = row.get(km_key)
            raw_price = row.get(price_key)
            if raw_km is None or raw_price is None:
                raise ValueError(f"Missing column value at line {line_no}: {row}")

            km = _parse_finite_float(raw_km, field="km", line_no=line_no)
            price = _parse_finite_float(raw_price, field="price", line_no=line_no)
            mileages.append(km)
            prices.append(price)

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def validate_dataset(mileages: list[float], prices: list[float]) -> None:
    if not mileages or not prices:
        raise ValueError("Empty dataset.")
    if len(mileages) != len(prices):
        raise ValueError("Mismatched dataset sizes.")
    _validate_finite_series("mileage", mileages)
    _validate_finite_series("price", prices)

    mileage_span = max(mileages) - min(mileages)
    if mileage_span < EPSILON:
        raise ValueError("Not enough variance in mileage.")


def compute_errors(prices: list[float], predictions: list[float]) -> ErrorStats:
    if not prices:
        raise ValueError("Empty dataset: cannot compute metrics.")
    if len(prices) != len(predictions):
        raise ValueError("Mismatched prediction and target sizes.")

    residuals: list[float] = []
    abs_error_sum = 0.0
    sq_error_sum = 0.0
    max_abs_error = 0.0

    for actual, predicted in zip(prices, predictions):
        residual = predicted - actual
        abs_residual = abs(residual)
        sq_residual = residual * residual

        residuals.append(residual)
        abs_error_sum += abs_residual
        sq_error_sum += sq_residual
        if abs_residual > max_abs_error:
            max_abs_error = abs_residual

    return ErrorStats(
        residuals=tuple(residuals),
        abs_error_sum=abs_error_sum,
        sq_error_sum=sq_error_sum,
        max_abs_error=max_abs_error,
    )


def compute_variance_stats(residuals: tuple[float, ...], sq_error_sum: float) -> tuple[float, float, int]:
    sample_count = len(residuals)
    if sample_count == 0:
        raise ValueError("Cannot compute variance stats on empty residuals.")

    mean_error = sum(residuals) / sample_count
    error_variance = (sq_error_sum / sample_count) - (mean_error * mean_error)
    if error_variance < 0.0:
        error_variance = 0.0
    error_std = error_variance**0.5

    outlier_count = 0
    if error_std > 0.0:
        for residual in residuals:
            z_score = (residual - mean_error) / error_std
            if abs(z_score) > OUTLIER_Z_THRESHOLD:
                outlier_count += 1

    return mean_error, error_std, outlier_count


def compute_r2(prices: list[float], sq_error_sum: float) -> tuple[float | None, bool]:
    if not prices:
        raise ValueError("Empty dataset: cannot compute R2.")

    y_mean = mean(prices)
    ss_tot = 0.0
    for value in prices:
        centered = value - y_mean
        ss_tot += centered * centered

    if ss_tot < EPSILON:
        return None, False
    return 1.0 - (sq_error_sum / ss_tot), True


def compute_metrics(prices: list[float], predictions: list[float]) -> RegressionMetrics:
    errors = compute_errors(prices, predictions)
    sample_count = len(prices)
    mse = errors.sq_error_sum / sample_count
    mean_error, error_std, outlier_count = compute_variance_stats(
        errors.residuals, errors.sq_error_sum
    )
    r2, r2_defined = compute_r2(prices, errors.sq_error_sum)

    return RegressionMetrics(
        mae=errors.abs_error_sum / sample_count,
        mse=mse,
        rmse=mse**0.5,
        r2=r2,
        r2_defined=r2_defined,
        mean_error=mean_error,
        error_std=error_std,
        max_abs_error=errors.max_abs_error,
        outlier_count=outlier_count,
    )


def compare_with_baseline(prices: list[float], model_predictions: list[float]) -> MetricsComparison:
    model_metrics = compute_metrics(prices, model_predictions)

    baseline_value = mean(prices)
    baseline_predictions = [baseline_value] * len(prices)
    baseline_metrics = compute_metrics(prices, baseline_predictions)

    delta_mse = baseline_metrics.mse - model_metrics.mse

    signal_to_noise_ratio: float | None
    if model_metrics.mse <= EPSILON:
        signal_to_noise_ratio = None
    else:
        signal_to_noise_ratio = baseline_metrics.mse / model_metrics.mse

    usefulness_score: float | None
    if baseline_metrics.mse <= EPSILON:
        usefulness_score = None
    else:
        usefulness_score = delta_mse / baseline_metrics.mse

    return MetricsComparison(
        model=model_metrics,
        baseline=baseline_metrics,
        delta_mse=delta_mse,
        signal_to_noise_ratio=signal_to_noise_ratio,
        usefulness_score=usefulness_score,
    )


def correlation(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y):
        raise ValueError("Mismatched series sizes for correlation.")
    if not values_x:
        raise ValueError("Empty series for correlation.")

    x_mean = mean(values_x)
    y_mean = mean(values_y)

    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0

    for x_value, y_value in zip(values_x, values_y):
        x_centered = x_value - x_mean
        y_centered = y_value - y_mean
        covariance += x_centered * y_centered
        variance_x += x_centered * x_centered
        variance_y += y_centered * y_centered

    if variance_x < EPSILON or variance_y < EPSILON:
        return None
    return covariance / ((variance_x * variance_y) ** 0.5)


def evaluate(
    mileages: list[float],
    prices: list[float],
    model: Model,
    test_ratio: float,
    seed: int,
) -> EvaluationResult:
    validate_dataset(mileages, prices)

    full_predictions = predict(model, mileages)
    full_comparison = compare_with_baseline(prices, full_predictions)

    train_mileages, train_prices, test_mileages, test_prices = split_dataset(
        mileages, prices, test_ratio=test_ratio, seed=seed
    )
    train_predictions = predict(model, train_mileages)
    test_predictions = predict(model, test_mileages)

    train_comparison = compare_with_baseline(train_prices, train_predictions)
    test_comparison = compare_with_baseline(test_prices, test_predictions)

    return EvaluationResult(
        samples=len(prices),
        full=full_comparison,
        train=train_comparison,
        test=test_comparison,
        split=SplitInfo(
            test_ratio=test_ratio,
            seed=seed,
            train_samples=len(train_mileages),
            test_samples=len(test_mileages),
        ),
        mileage_price_correlation=correlation(mileages, prices),
    )


def _format_r2(value: float | None, defined: bool) -> str:
    return "undefined (constant target values)" if not defined else f"{value:.6f}"


def _format_optional(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def serialize_result(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)


def render_text_output(result: EvaluationResult) -> str:
    lines: list[str] = [
        f"Samples      : {result.samples}",
        (
            "Split        : "
            f"{result.split.train_samples} train / {result.split.test_samples} test "
            f"(ratio={result.split.test_ratio}, seed={result.split.seed})"
        ),
    ]

    def append_comparison(title: str, comparison: MetricsComparison) -> None:
        lines.append(title)
        lines.append(f"MAE          : {comparison.model.mae:.6f}")
        lines.append(f"MSE          : {comparison.model.mse:.6f}")
        lines.append(f"RMSE         : {comparison.model.rmse:.6f}")
        lines.append(f"R2           : {_format_r2(comparison.model.r2, comparison.model.r2_defined)}")
        lines.append(f"Mean Error   : {comparison.model.mean_error:.6f}")
        lines.append(f"Error Std    : {comparison.model.error_std:.6f}")
        lines.append(f"Max Abs Error: {comparison.model.max_abs_error:.6f}")
        lines.append(f"Outliers     : {comparison.model.outlier_count}")
        lines.append("Baseline metrics (predict mean(price))")
        lines.append(f"MAE          : {comparison.baseline.mae:.6f}")
        lines.append(f"MSE          : {comparison.baseline.mse:.6f}")
        lines.append(f"RMSE         : {comparison.baseline.rmse:.6f}")
        lines.append(
            "R2           : "
            f"{_format_r2(comparison.baseline.r2, comparison.baseline.r2_defined)}"
        )
        lines.append(f"Mean Error   : {comparison.baseline.mean_error:.6f}")
        lines.append(f"Error Std    : {comparison.baseline.error_std:.6f}")
        lines.append(f"Max Abs Error: {comparison.baseline.max_abs_error:.6f}")
        lines.append(f"Outliers     : {comparison.baseline.outlier_count}")
        lines.append(f"Delta MSE    : {comparison.delta_mse:.6f} (positive means model is better)")
        lines.append(f"SNR          : {_format_optional(comparison.signal_to_noise_ratio)}")
        lines.append(f"Usefulness   : {_format_optional(comparison.usefulness_score)}")

    append_comparison("Full dataset metrics", result.full)
    append_comparison("Train split metrics", result.train)
    append_comparison("Test split metrics", result.test)
    if result.mileage_price_correlation is None:
        lines.append("Correlation  : undefined")
    else:
        lines.append(f"Correlation  : {result.mileage_price_correlation:.6f}")

    return "\n".join(lines)


def emit_output(result: EvaluationResult, json_output: bool) -> None:
    if json_output:
        print(json.dumps(serialize_result(result), indent=2))
        return
    print(render_text_output(result))


def save_report(path: Path, result: EvaluationResult) -> None:
    path.write_text(json.dumps(serialize_result(result), indent=2), encoding="utf-8")


def parse_args() -> EvaluateArgs:
    parser = argparse.ArgumentParser(description="Evaluate model performance on a dataset.")
    parser.add_argument("--dataset", default="data.csv", help="Dataset CSV (default: data.csv)")
    parser.add_argument("--model", default="model.json", help="Model JSON (default: model.json)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output metrics as JSON.")
    parser.add_argument("--report", help="Optional path to save evaluation report JSON.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio in (0, 1) (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/test split (default: 42)",
    )
    ns = parser.parse_args()
    if not 0.0 < ns.test_ratio < 1.0:
        parser.error("--test-ratio must be in the interval (0, 1).")
    return EvaluateArgs(
        dataset_path=Path(ns.dataset),
        model_path=Path(ns.model),
        json_output=ns.json_output,
        report_path=Path(ns.report) if ns.report else None,
        test_ratio=ns.test_ratio,
        seed=ns.seed,
    )


def log_comparison_warnings(logger: logging.Logger, scope_name: str, comparison: MetricsComparison) -> None:
    if not comparison.model.r2_defined:
        logger.warning(
            "Warning: %s model R2 is undefined because target values are constant.",
            scope_name,
        )
    if not comparison.baseline.r2_defined:
        logger.warning(
            "Warning: %s baseline R2 is undefined because target values are constant.",
            scope_name,
        )
    if comparison.model.mse > comparison.baseline.mse:
        logger.warning("Warning: %s model MSE is worse than baseline MSE.", scope_name)


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        mileages, prices = load_dataset(args.dataset_path)
        model, _ = load_model(args.model_path, model_policy=ModelPolicy.STRICT, logger=logger)
        result = evaluate(
            mileages,
            prices,
            model,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    for scope_name, comparison in (("full", result.full), ("train", result.train), ("test", result.test)):
        log_comparison_warnings(logger, scope_name, comparison)

    if args.report_path is not None:
        try:
            save_report(args.report_path, result)
        except OSError as exc:
            logger.error("Error: cannot save report to %s (%s)", args.report_path, exc)
            raise SystemExit(1)

    emit_output(result, json_output=args.json_output)


if __name__ == "__main__":
    main()
