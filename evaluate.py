#!/usr/bin/env python3
"""Evaluate trained model quality on a CSV dataset."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
from pathlib import Path

from predictor.engine import load_model, predict
from predictor.model import Model, ModelPolicy

LOGGER_NAME = "ft_linear_regression.evaluate"
EPSILON = 1e-12
R2_UNDEFINED: None = None


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    samples: int
    model: RegressionMetrics
    baseline: RegressionMetrics
    mileage_price_correlation: float | None


@dataclasses.dataclass(frozen=True)
class EvaluateArgs:
    dataset_path: Path
    model_path: Path
    json_output: bool
    report_path: Path | None


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    return logging.getLogger(LOGGER_NAME)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Empty dataset: cannot compute mean.")
    return sum(values) / len(values)


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

    return mileages, prices


def validate_dataset(mileages: list[float], prices: list[float]) -> None:
    """Single source of truth for dataset validation in evaluation flow."""
    if not mileages or not prices:
        raise ValueError("Empty dataset.")
    if len(mileages) != len(prices):
        raise ValueError("Mismatched dataset sizes.")

    for index, value in enumerate(mileages):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite mileage at index {index}.")
    for index, value in enumerate(prices):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite price at index {index}.")

    mileage_span = max(mileages) - min(mileages)
    if mileage_span < EPSILON:
        raise ValueError("Not enough variance in mileage.")


def _compute_metrics(prices: list[float], predictions: list[float]) -> RegressionMetrics:
    if not prices:
        raise ValueError("Empty dataset: cannot compute metrics.")
    if len(prices) != len(predictions):
        raise ValueError("Mismatched prediction and target sizes.")

    m = len(prices)

    abs_error_sum = 0.0
    sq_error_sum = 0.0
    error_sum = 0.0
    price_sum = 0.0
    price_sq_sum = 0.0
    max_abs_error = 0.0

    for idx in range(m):
        actual = prices[idx]
        predicted = predictions[idx]
        error = predicted - actual
        abs_error = abs(error)
        sq_error = error * error

        abs_error_sum += abs_error
        sq_error_sum += sq_error
        error_sum += error
        price_sum += actual
        price_sq_sum += actual * actual
        if abs_error > max_abs_error:
            max_abs_error = abs_error

    mse = sq_error_sum / m
    mean_error = error_sum / m
    error_variance = (sq_error_sum / m) - (mean_error * mean_error)
    if error_variance < 0.0:
        error_variance = 0.0
    error_std = error_variance**0.5

    outlier_threshold = 3.0 * error_std
    outlier_count = 0
    if outlier_threshold > 0.0:
        for idx in range(m):
            residual = predictions[idx] - prices[idx]
            if abs(residual) > outlier_threshold:
                outlier_count += 1

    r2: float | None
    r2_defined: bool
    ss_tot = price_sq_sum - ((price_sum * price_sum) / m)
    if ss_tot < 0.0 and abs(ss_tot) < EPSILON:
        ss_tot = 0.0
    if ss_tot < EPSILON:
        r2 = R2_UNDEFINED
        r2_defined = False
    else:
        r2 = 1.0 - (sq_error_sum / ss_tot)
        r2_defined = True

    return RegressionMetrics(
        mae=abs_error_sum / m,
        mse=mse,
        rmse=mse**0.5,
        r2=r2,
        r2_defined=r2_defined,
        mean_error=mean_error,
        error_std=error_std,
        max_abs_error=max_abs_error,
        outlier_count=outlier_count,
    )


def correlation(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y):
        raise ValueError("Mismatched series sizes for correlation.")
    if not values_x:
        raise ValueError("Empty series for correlation.")

    n = len(values_x)
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    for idx in range(n):
        x = values_x[idx]
        y = values_y[idx]
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y

    numerator = (n * sum_xy) - (sum_x * sum_y)
    denom_left = (n * sum_x2) - (sum_x * sum_x)
    denom_right = (n * sum_y2) - (sum_y * sum_y)
    if denom_left < EPSILON or denom_right < EPSILON:
        return None
    denominator = (denom_left * denom_right) ** 0.5
    if denominator < EPSILON:
        return None
    return numerator / denominator


def evaluate(mileages: list[float], prices: list[float], model: Model) -> EvaluationResult:
    validate_dataset(mileages, prices)

    predictions = predict(model, mileages)
    model_metrics = _compute_metrics(prices, predictions)

    baseline_value = mean(prices)
    baseline_predictions = [baseline_value] * len(prices)
    baseline_metrics = _compute_metrics(prices, baseline_predictions)

    return EvaluationResult(
        samples=len(prices),
        model=model_metrics,
        baseline=baseline_metrics,
        mileage_price_correlation=correlation(mileages, prices),
    )


def _serialize_metrics(metrics: RegressionMetrics) -> dict[str, float | int | bool | None]:
    return {
        "mae": metrics.mae,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "r2_defined": metrics.r2_defined,
        "mean_error": metrics.mean_error,
        "error_std": metrics.error_std,
        "max_abs_error": metrics.max_abs_error,
        "outlier_count": metrics.outlier_count,
    }


def build_report_payload(result: EvaluationResult) -> dict[str, object]:
    return {
        "samples": result.samples,
        "model": _serialize_metrics(result.model),
        "baseline": _serialize_metrics(result.baseline),
        "delta_mse": result.baseline.mse - result.model.mse,
        "mileage_price_correlation": result.mileage_price_correlation,
    }


def _format_r2(value: float | None, defined: bool) -> str:
    return "undefined (constant target values)" if not defined else f"{value:.6f}"


def emit_output(payload: dict[str, object], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return

    model_metrics = payload["model"]
    baseline_metrics = payload["baseline"]

    if not isinstance(model_metrics, dict) or not isinstance(baseline_metrics, dict):
        raise ValueError("Invalid report payload shape.")

    print(f"Samples      : {payload['samples']}")
    print("Model metrics")
    print(f"MAE          : {float(model_metrics['mae']):.6f}")
    print(f"MSE          : {float(model_metrics['mse']):.6f}")
    print(f"RMSE         : {float(model_metrics['rmse']):.6f}")
    print(
        "R2           : "
        f"{_format_r2(model_metrics.get('r2'), bool(model_metrics.get('r2_defined')))}"
    )
    print(f"Mean Error   : {float(model_metrics['mean_error']):.6f}")
    print(f"Error Std    : {float(model_metrics['error_std']):.6f}")
    print(f"Max Abs Error: {float(model_metrics['max_abs_error']):.6f}")
    print(f"Outliers     : {int(model_metrics['outlier_count'])}")
    print("Baseline metrics (predict mean(price))")
    print(f"MAE          : {float(baseline_metrics['mae']):.6f}")
    print(f"MSE          : {float(baseline_metrics['mse']):.6f}")
    print(f"RMSE         : {float(baseline_metrics['rmse']):.6f}")
    print(
        "R2           : "
        f"{_format_r2(baseline_metrics.get('r2'), bool(baseline_metrics.get('r2_defined')))}"
    )
    print(f"Mean Error   : {float(baseline_metrics['mean_error']):.6f}")
    print(f"Error Std    : {float(baseline_metrics['error_std']):.6f}")
    print(f"Max Abs Error: {float(baseline_metrics['max_abs_error']):.6f}")
    print(f"Outliers     : {int(baseline_metrics['outlier_count'])}")
    print(
        f"Delta MSE    : {float(payload['delta_mse']):.6f} "
        "(positive means model is better)"
    )
    correlation_value = payload.get("mileage_price_correlation")
    if correlation_value is None:
        print("Correlation  : undefined")
    else:
        print(f"Correlation  : {float(correlation_value):.6f}")


def save_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> EvaluateArgs:
    parser = argparse.ArgumentParser(description="Evaluate model performance on a dataset.")
    parser.add_argument("--dataset", default="data.csv", help="Dataset CSV (default: data.csv)")
    parser.add_argument("--model", default="model.json", help="Model JSON (default: model.json)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output metrics as JSON.")
    parser.add_argument(
        "--report",
        help="Optional path to save evaluation report JSON.",
    )
    ns = parser.parse_args()
    return EvaluateArgs(
        dataset_path=Path(ns.dataset),
        model_path=Path(ns.model),
        json_output=ns.json_output,
        report_path=Path(ns.report) if ns.report else None,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        mileages, prices = load_dataset(args.dataset_path)
        model, _ = load_model(args.model_path, model_policy=ModelPolicy.STRICT, logger=logger)
        result = evaluate(mileages, prices, model)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    if not result.model.r2_defined:
        logger.warning("Warning: model R2 is undefined because all target values are constant.")
    if not result.baseline.r2_defined:
        logger.warning("Warning: baseline R2 is undefined because all target values are constant.")
    if result.model.mse > result.baseline.mse:
        logger.warning("Warning: model MSE is worse than baseline MSE.")

    payload = build_report_payload(result)

    if args.report_path is not None:
        try:
            save_report(args.report_path, payload)
        except OSError as exc:
            logger.error("Error: cannot save report to %s (%s)", args.report_path, exc)
            raise SystemExit(1)

    emit_output(payload, json_output=args.json_output)


if __name__ == "__main__":
    main()
