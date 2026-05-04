from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .model import LOGGER_NAME, TrainArgs
from .pipeline import run_pipeline


def configure_logging(verbosity: str) -> logging.Logger:
    levels = {
        "quiet": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    logging.basicConfig(level=levels[verbosity], format="%(message)s", force=True)
    return logging.getLogger(LOGGER_NAME)


def parse_args() -> TrainArgs:
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
        "--early-stopping-patience",
        type=int,
        default=300,
        help="Stop if train loss does not improve for N iterations (default: 300)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-6,
        help="Minimum train-loss improvement to reset patience (default: 1e-6)",
    )
    parser.add_argument(
        "--verbosity",
        choices=("quiet", "info", "debug"),
        default="info",
        help="CLI verbosity level (default: info)",
    )
    ns = parser.parse_args()
    return TrainArgs(
        dataset_path=Path(ns.dataset),
        model_path=Path(ns.model),
        learning_rate=ns.learning_rate,
        iterations=ns.iterations,
        test_ratio=ns.test_ratio,
        seed=ns.seed,
        log_every=ns.log_every,
        early_stopping_patience=ns.early_stopping_patience,
        early_stopping_min_delta=ns.early_stopping_min_delta,
        verbosity=ns.verbosity,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.verbosity)
    try:
        run_pipeline(args, logger)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)
