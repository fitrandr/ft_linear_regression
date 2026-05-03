#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

EPSILON = 1e-12
LOGGER_NAME = "ft_linear_regression.train"

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
        return estimate_price(mileage, self.theta0, self.theta1)


@dataclass(frozen=True)
class TrainingLogEntry:
    iteration: int
    mse: float


@dataclass(frozen=True)
class TrainingResult:
    model: LinearModel
    learning_rate: float
    iterations: int
    log_every: int
    samples: int
    history: tuple[TrainingLogEntry, ...]


@dataclass(frozen=True)
class EvaluationMetrics:
    train_mse: float
    test_mse: float


def configure_logging(verbosity: str) -> logging.Logger:
    """Configure logger for CLI output verbosity."""
    levels = {
        "quiet": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    logging.basicConfig(
        level=levels[verbosity],
        format="%(message)s",
        force=True,
    )
    return logging.getLogger(LOGGER_NAME)


def load_dataset(path: Path) -> tuple[list[float], list[float]]:
    """Load dataset from CSV and validate numeric values."""
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
            raw_km = (row.get(km_key) or "").strip()
            raw_price = (row.get(price_key) or "").strip()

            if raw_km == "" or raw_price == "":
                raise ValueError(f"Missing value at line {line_no}: {row}")

            try:
                km = float(raw_km)
                price = float(raw_price)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value at line {line_no}: {row}") from exc

            if not math.isfinite(km) or not math.isfinite(price):
                raise ValueError(f"Non-finite numeric value at line {line_no}: {row}")

            mileages.append(km)
            prices.append(price)

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def validate_pairs(mileages: list[float], prices: list[float]) -> None:
    """Validate feature/target pairing constraints."""
    if not mileages or not prices:
        raise ValueError("mileages and prices must not be empty.")
    if len(mileages) != len(prices):
        raise ValueError("mileages and prices must have the same number of samples.")


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(values) / len(values)


def std(values: list[float], values_mean: float) -> float:
    """Return population standard deviation."""
    if not values:
        raise ValueError("Cannot compute std of an empty list.")
    variance = sum((value - values_mean) ** 2 for value in values) / len(values)
    return variance**0.5


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Linear regression hypothesis: y_hat = theta0 + theta1 * x."""
    return theta0 + theta1 * mileage


def predict(model: LinearModel, mileage: float) -> float:
    """Predict price from a typed model object."""
    return model.predict(mileage)


def mse(mileages: list[float], prices: list[float], model: LinearModel) -> float:
    """Compute Mean Squared Error."""
    validate_pairs(mileages, prices)
    m = len(mileages)
    total = 0.0
    for km, price in zip(mileages, prices):
        error = predict(model, km) - price
        total += error * error
    return total / m


def split_dataset(
    mileages: list[float],
    prices: list[float],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Split dataset into train/test subsets."""
    validate_pairs(mileages, prices)
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in the interval (0, 1).")

    m = len(mileages)
    if m < 2:
        raise ValueError("At least 2 samples are required for train/test split.")

    indices = list(range(m))
    random.Random(seed).shuffle(indices)

    test_size = max(1, int(m * test_ratio))
    if test_size >= m:
        test_size = m - 1

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_mileages = [mileages[i] for i in train_indices]
    train_prices = [prices[i] for i in train_indices]
    test_mileages = [mileages[i] for i in test_indices]
    test_prices = [prices[i] for i in test_indices]
    return train_mileages, train_prices, test_mileages, test_prices


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float,
    iterations: int,
    log_every: int,
    logger: logging.Logger,
) -> TrainingResult:
    """Train linear regression with gradient descent."""
    validate_pairs(mileages, prices)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")
    if log_every < 0:
        raise ValueError("log_every must be >= 0.")

    m = len(mileages)
    km_mean = mean(mileages)
    km_std = std(mileages, km_mean)
    if km_std < EPSILON:
        raise ValueError("Mileage variance is too small; cannot train a stable slope.")

    norm_mileages = [(km - km_mean) / km_std for km in mileages]
    theta0 = 0.0
    theta1 = 0.0
    history: list[TrainingLogEntry] = []

    for iteration in range(1, iterations + 1):
        gradient0 = 0.0
        gradient1 = 0.0
        squared_error_sum = 0.0

        for mileage, price in zip(norm_mileages, prices):
            error = estimate_price(mileage, theta0, theta1) - price
            gradient0 += error
            gradient1 += error * mileage
            squared_error_sum += error * error

        should_log = (
            log_every > 0
            and (iteration == 1 or iteration == iterations or iteration % log_every == 0)
        )
        if should_log:
            train_mse = squared_error_sum / m
            history.append(TrainingLogEntry(iteration=iteration, mse=train_mse))
            logger.info("Iteration %d/%d - Train MSE: %.6f", iteration, iterations, train_mse)

        theta0 -= learning_rate * (gradient0 / m)
        theta1 -= learning_rate * (gradient1 / m)

    model = LinearModel(
        theta0=theta0 - (theta1 * km_mean / km_std),
        theta1=theta1 / km_std,
        normalized_theta0=theta0,
        normalized_theta1=theta1,
        km_mean=km_mean,
        km_std=km_std,
    )

    return TrainingResult(
        model=model,
        learning_rate=learning_rate,
        iterations=iterations,
        log_every=log_every,
        samples=m,
        history=tuple(history),
    )


def train_pipeline(
    dataset_path: Path,
    learning_rate: float,
    iterations: int,
    test_ratio: float,
    seed: int,
    log_every: int,
    logger: logging.Logger,
) -> tuple[TrainingResult, DatasetSplit]:
    """Load data, split, and train model."""
    mileages, prices = load_dataset(dataset_path)
    split = split_dataset(mileages, prices, test_ratio=test_ratio, seed=seed)
    train_mileages, train_prices, test_mileages, _ = split
    logger.debug("Dataset loaded: %d samples", len(mileages))
    logger.info(
        "Dataset split complete: %d train / %d test",
        len(train_mileages),
        len(test_mileages),
    )

    result = train(
        train_mileages,
        train_prices,
        learning_rate=learning_rate,
        iterations=iterations,
        log_every=log_every,
        logger=logger,
    )
    return result, split


def evaluate_pipeline(model: LinearModel, split: DatasetSplit) -> EvaluationMetrics:
    """Compute train/test evaluation metrics for a trained model."""
    train_mileages, train_prices, test_mileages, test_prices = split
    return EvaluationMetrics(
        train_mse=mse(train_mileages, train_prices, model),
        test_mse=mse(test_mileages, test_prices, model),
    )


def build_payload(
    training_result: TrainingResult,
    metrics: EvaluationMetrics,
    split: DatasetSplit,
    test_ratio: float,
    seed: int,
) -> dict[str, object]:
    """Build JSON-serializable payload for model persistence."""
    train_mileages, _, test_mileages, _ = split
    model_payload = asdict(training_result.model)
    history_payload = [asdict(entry) for entry in training_result.history]

    payload: dict[str, object] = {
        # Backward-compatible top-level model parameters.
        "theta0": training_result.model.theta0,
        "theta1": training_result.model.theta1,
        "km_mean": training_result.model.km_mean,
        "km_std": training_result.model.km_std,
        "normalized_theta0": training_result.model.normalized_theta0,
        "normalized_theta1": training_result.model.normalized_theta1,
        # Structured sections.
        "model": model_payload,
        "training": {
            "learning_rate": training_result.learning_rate,
            "iterations": training_result.iterations,
            "samples": training_result.samples,
            "log_every": training_result.log_every,
            "train_samples": len(train_mileages),
            "test_samples": len(test_mileages),
            "test_ratio": test_ratio,
            "seed": seed,
        },
        "metrics": asdict(metrics),
        "history": history_payload,
    }
    return payload


def save_model(model_path: Path, payload: dict[str, object]) -> None:
    """Serialize and save trained model payload."""
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train linear regression on data.csv")
    parser.add_argument(
        "--dataset",
        default="data.csv",
        help="Path to CSV dataset (default: data.csv)",
    )
    parser.add_argument(
        "--model",
        default="model.json",
        help="Path to output model file (default: model.json)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Gradient descent learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of training iterations (default: 10000)",
    )
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log train MSE every N iterations (0 disables progress logs)",
    )
    parser.add_argument(
        "--verbosity",
        choices=("quiet", "info", "debug"),
        default="info",
        help="CLI verbosity level (default: info)",
    )
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and save linear regression model."""
    args = parse_args()
    logger = configure_logging(args.verbosity)

    try:
        model_path = Path(args.model)
        training_result, split = train_pipeline(
            dataset_path=Path(args.dataset),
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            test_ratio=args.test_ratio,
            seed=args.seed,
            log_every=args.log_every,
            logger=logger,
        )
        metrics = evaluate_pipeline(training_result.model, split)
        payload = build_payload(
            training_result=training_result,
            metrics=metrics,
            split=split,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        save_model(model_path, payload)

        train_mileages, _, test_mileages, _ = split
        logger.info("Training complete on %d train samples.", len(train_mileages))
        logger.info("Evaluation complete on %d test samples.", len(test_mileages))
        logger.info("theta0    = %.6f", training_result.model.theta0)
        logger.info("theta1    = %.6f", training_result.model.theta1)
        logger.info("Train MSE = %.6f", metrics.train_mse)
        logger.info("Test MSE  = %.6f", metrics.test_mse)
        logger.info("Model saved to %s", model_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
