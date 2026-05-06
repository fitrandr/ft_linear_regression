from __future__ import annotations

import argparse
import logging
from pathlib import Path

from predictor.engine import load_model
from predictor.model import ModelPolicy

from .data import load_dataset
from .model import EvaluateArgs, LOGGER_NAME
from .pipeline import evaluate, log_comparison_warnings
from .report import emit_output, save_report


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


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
