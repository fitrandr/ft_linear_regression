#!/usr/bin/env python3
"""Evaluate trained model quality on a CSV dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from predictor.engine import load_model, predict
from predictor.model import Model, ModelPolicy
from trainer.data import load_dataset, validate_pairs

LOGGER_NAME = "ft_linear_regression.evaluate"
EPSILON = 1e-12


@dataclasses.dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    mse: float
    rmse: float
    r2: float | None


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    samples: int
    model: RegressionMetrics
    baseline: RegressionMetrics


@dataclasses.dataclass(frozen=True)
class EvaluateArgs:
    dataset_path: Path
    model_path: Path
    json_output: bool


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
    return logging.getLogger(LOGGER_NAME)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Empty dataset: cannot compute mean.")
    return sum(values) / len(values)


def _compute_r2(prices: list[float], predictions: list[float]) -> float | None:
    y_mean = mean(prices)
    ss_tot = sum((actual - y_mean) ** 2 for actual in prices)
    if ss_tot < EPSILON:
        return None
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(prices, predictions))
    return 1.0 - (ss_res / ss_tot)


def _compute_metrics(prices: list[float], predictions: list[float]) -> RegressionMetrics:
    if not prices:
        raise ValueError("Empty dataset: cannot compute metrics.")
    if len(prices) != len(predictions):
        raise ValueError("Mismatched prediction and target sizes.")

    errors = [pred - actual for pred, actual in zip(predictions, prices)]
    abs_errors = [abs(err) for err in errors]
    sq_errors = [err * err for err in errors]

    m = len(prices)
    mse = sum(sq_errors) / m
    return RegressionMetrics(
        mae=sum(abs_errors) / m,
        mse=mse,
        rmse=mse**0.5,
        r2=_compute_r2(prices, predictions),
    )


def evaluate_dataset(mileages: list[float], prices: list[float], model: Model) -> EvaluationResult:
    validate_pairs(mileages, prices)

    predictions = predict(model, mileages)
    model_metrics = _compute_metrics(prices, predictions)

    baseline_value = mean(prices)
    baseline_predictions = [baseline_value] * len(prices)
    baseline_metrics = _compute_metrics(prices, baseline_predictions)

    return EvaluationResult(
        samples=len(prices),
        model=model_metrics,
        baseline=baseline_metrics,
    )


def _serialize_metrics(metrics: RegressionMetrics) -> dict[str, float | None]:
    return {
        "mae": metrics.mae,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
    }


def emit_output(result: EvaluationResult, json_output: bool) -> None:
    if json_output:
        payload = {
            "samples": result.samples,
            "model": _serialize_metrics(result.model),
            "baseline": _serialize_metrics(result.baseline),
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Samples      : {result.samples}")
    print("Model metrics")
    print(f"MAE          : {result.model.mae:.6f}")
    print(f"MSE          : {result.model.mse:.6f}")
    print(f"RMSE         : {result.model.rmse:.6f}")
    print("R2           : undefined (constant target values)" if result.model.r2 is None else f"R2           : {result.model.r2:.6f}")
    print("Baseline metrics (predict mean(price))")
    print(f"MAE          : {result.baseline.mae:.6f}")
    print(f"MSE          : {result.baseline.mse:.6f}")
    print(f"RMSE         : {result.baseline.rmse:.6f}")
    print("R2           : undefined (constant target values)" if result.baseline.r2 is None else f"R2           : {result.baseline.r2:.6f}")
    print(f"Delta MSE    : {result.baseline.mse - result.model.mse:.6f} (positive means model is better)")


def parse_args() -> EvaluateArgs:
    parser = argparse.ArgumentParser(description="Evaluate model performance on a dataset.")
    parser.add_argument("--dataset", default="data.csv", help="Dataset CSV (default: data.csv)")
    parser.add_argument("--model", default="model.json", help="Model JSON (default: model.json)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output metrics as JSON.")
    ns = parser.parse_args()
    return EvaluateArgs(
        dataset_path=Path(ns.dataset),
        model_path=Path(ns.model),
        json_output=ns.json_output,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        mileages, prices = load_dataset(args.dataset_path)
        validate_pairs(mileages, prices)
        model, _ = load_model(args.model_path, model_policy=ModelPolicy.STRICT, logger=logger)
        result = evaluate_dataset(mileages, prices, model)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    if result.model.r2 is None:
        logger.warning("Warning: model R2 is undefined because all target values are constant.")
    if result.baseline.r2 is None:
        logger.warning("Warning: baseline R2 is undefined because all target values are constant.")

    emit_output(result, json_output=args.json_output)


if __name__ == "__main__":
    main()
