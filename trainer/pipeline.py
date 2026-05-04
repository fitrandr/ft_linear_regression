from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .data import load_dataset, split_dataset
from .engine import mse, train_model
from .model import (
    MODEL_VERSION,
    DatasetSplit,
    EvaluationMetrics,
    PipelineOutput,
    TrainArgs,
    TrainingResult,
)


def evaluate_model(training_result: TrainingResult, split: DatasetSplit) -> EvaluationMetrics:
    train_mileages, train_prices, test_mileages, test_prices = split
    return EvaluationMetrics(
        train_mse=mse(train_mileages, train_prices, training_result.model),
        test_mse=mse(test_mileages, test_prices, training_result.model),
    )


def build_payload(
    training_result: TrainingResult,
    metrics: EvaluationMetrics,
    split: DatasetSplit,
    test_ratio: float,
    seed: int,
) -> dict[str, object]:
    train_mileages, _, test_mileages, _ = split
    model_payload = asdict(training_result.model)
    history_payload = [asdict(entry) for entry in training_result.history]

    payload: dict[str, object] = {
        "version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
            "iterations_ran": training_result.iterations_ran,
            "samples": training_result.samples,
            "log_every": training_result.log_every,
            "stopped_early": training_result.stopped_early,
            "best_train_mse": training_result.best_train_mse,
            "early_stopping_patience": training_result.early_stopping_patience,
            "early_stopping_min_delta": training_result.early_stopping_min_delta,
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
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(args: TrainArgs, logger: logging.Logger) -> PipelineOutput:
    mileages, prices = load_dataset(args.dataset_path)
    split = split_dataset(mileages, prices, test_ratio=args.test_ratio, seed=args.seed)
    train_mileages, train_prices, test_mileages, _ = split
    logger.debug("Dataset loaded: %d samples", len(mileages))
    logger.info(
        "Dataset split complete: %d train / %d test",
        len(train_mileages),
        len(test_mileages),
    )

    training_result = train_model(
        train_mileages,
        train_prices,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        log_every=args.log_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        logger=logger,
    )
    metrics = evaluate_model(training_result, split)
    payload = build_payload(
        training_result=training_result,
        metrics=metrics,
        split=split,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_model(args.model_path, payload)

    logger.info("Training complete on %d train samples.", len(train_mileages))
    logger.info("Evaluation complete on %d test samples.", len(test_mileages))
    logger.info("theta0    = %.6f", training_result.model.theta0)
    logger.info("theta1    = %.6f", training_result.model.theta1)
    logger.info("Train MSE = %.6f", metrics.train_mse)
    logger.info("Test MSE  = %.6f", metrics.test_mse)
    logger.info("Model saved to %s", args.model_path)

    return PipelineOutput(
        model_path=args.model_path,
        training_result=training_result,
        metrics=metrics,
        split=split,
    )
