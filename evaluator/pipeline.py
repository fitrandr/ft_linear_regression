from __future__ import annotations

import logging

from predictor.engine import predict
from predictor.model import Model
from trainer.data import split_dataset

from .data import validate_dataset
from .model import EvaluationResult, MetricsComparison, SplitInfo
from .stats import compare_with_baseline, correlation


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
