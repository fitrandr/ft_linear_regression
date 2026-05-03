#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

EPSILON = 1e-12


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


def predict(model: dict[str, float], mileage: float) -> float:
    """Predict price from a saved model dict using raw-scale theta values."""
    return estimate_price(mileage, model["theta0"], model["theta1"])


def mse(mileages: list[float], prices: list[float], theta0: float, theta1: float) -> float:
    """Compute Mean Squared Error."""
    validate_pairs(mileages, prices)
    m = len(mileages)
    total = 0.0
    for km, price in zip(mileages, prices):
        error = estimate_price(km, theta0, theta1) - price
        total += error * error
    return total / m


def split_dataset(
    mileages: list[float],
    prices: list[float],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[float], list[float], list[float], list[float]]:
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
) -> dict[str, Any]:
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

    history: list[dict[str, float | int]] = []

    for iteration in range(1, iterations + 1):
        gradient0 = 0.0
        gradient1 = 0.0

        for mileage, price in zip(norm_mileages, prices):
            error = estimate_price(mileage, theta0, theta1) - price
            gradient0 += error
            gradient1 += error * mileage

        theta0 -= learning_rate * (gradient0 / m)
        theta1 -= learning_rate * (gradient1 / m)

        should_log = (
            log_every > 0
            and (iteration == 1 or iteration == iterations or iteration % log_every == 0)
        )
        if should_log:
            raw_theta1 = theta1 / km_std
            raw_theta0 = theta0 - (theta1 * km_mean / km_std)
            train_mse = mse(mileages, prices, raw_theta0, raw_theta1)
            history.append({"iteration": iteration, "mse": train_mse})
            print(f"Iteration {iteration}/{iterations} - Train MSE: {train_mse:.6f}")

    raw_theta1 = theta1 / km_std
    raw_theta0 = theta0 - (theta1 * km_mean / km_std)

    return {
        "model": {
            "theta0": raw_theta0,
            "theta1": raw_theta1,
            "normalized_theta0": theta0,
            "normalized_theta1": theta1,
            "km_mean": km_mean,
            "km_std": km_std,
        },
        "training": {
            "learning_rate": learning_rate,
            "iterations": iterations,
            "samples": m,
            "log_every": log_every,
        },
        "history": history,
    }


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
        help="Print train MSE every N iterations (0 disables logs)",
    )
    return parser.parse_args()


def main() -> None:
    """Train, evaluate (train/test), and save a regression model."""
    try:
        args = parse_args()
        dataset_path = Path(args.dataset)
        model_path = Path(args.model)

        mileages, prices = load_dataset(dataset_path)
        train_mileages, train_prices, test_mileages, test_prices = split_dataset(
            mileages,
            prices,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        training_result = train(
            train_mileages,
            train_prices,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            log_every=args.log_every,
        )
        model = training_result["model"]

        train_mse = mse(train_mileages, train_prices, model["theta0"], model["theta1"])
        test_mse = mse(test_mileages, test_prices, model["theta0"], model["theta1"])

        payload = {
            # Backward-compatible top-level model parameters.
            "theta0": model["theta0"],
            "theta1": model["theta1"],
            "km_mean": model["km_mean"],
            "km_std": model["km_std"],
            "normalized_theta0": model["normalized_theta0"],
            "normalized_theta1": model["normalized_theta1"],
            # Structured sections for cleaner architecture.
            "model": model,
            "training": {
                **training_result["training"],
                "train_samples": len(train_mileages),
                "test_samples": len(test_mileages),
                "test_ratio": args.test_ratio,
                "seed": args.seed,
            },
            "metrics": {
                "train_mse": train_mse,
                "test_mse": test_mse,
            },
            "history": training_result["history"],
        }

        model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"Training complete on {len(train_mileages)} train samples.")
        print(f"Evaluation complete on {len(test_mileages)} test samples.")
        print(f"theta0    = {model['theta0']:.6f}")
        print(f"theta1    = {model['theta1']:.6f}")
        print(f"Train MSE = {train_mse:.6f}")
        print(f"Test MSE  = {test_mse:.6f}")
        print(f"Model saved to {model_path}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
