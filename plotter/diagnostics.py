from __future__ import annotations

import math
import random
from dataclasses import dataclass

EPSILON = 1e-12


@dataclass(frozen=True)
class AnimationFrame:
    iteration: int
    theta0: float
    theta1: float
    mse: float


def build_test_flags(sample_count: int, test_ratio: float, seed: int) -> list[bool]:
    if sample_count < 2:
        raise ValueError("At least 2 samples are required for split diagnostics.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0,1).")

    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)

    test_size = max(1, int(sample_count * test_ratio))
    if test_size >= sample_count:
        test_size = sample_count - 1

    test_indices = set(indices[:test_size])
    return [idx in test_indices for idx in range(sample_count)]


def quality_label(r2: float | None, usefulness: float | None) -> str:
    if r2 is None or usefulness is None:
        return "undefined"
    if r2 >= 0.80 and usefulness >= 0.70:
        return "excellent"
    if r2 >= 0.60 and usefulness >= 0.40:
        return "good"
    if r2 >= 0.40 and usefulness >= 0.20:
        return "fair"
    if usefulness < 0.0:
        return "worse-than-baseline"
    return "weak"


def predict_minus_actual(predictions: list[float], prices: list[float]) -> list[float]:
    if len(predictions) != len(prices):
        raise ValueError("predictions and prices must have the same length.")
    return [pred - actual for pred, actual in zip(predictions, prices)]


def build_gradient_descent_frames(
    mileages: list[float],
    prices: list[float],
    iterations: int,
    learning_rate: float = 0.1,
) -> list[AnimationFrame]:
    if len(mileages) != len(prices):
        raise ValueError("mileages and prices must have the same length.")
    if not mileages:
        raise ValueError("Empty dataset for animation.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0 for animation.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0 for animation.")

    m = len(mileages)
    km_mean = sum(mileages) / m
    km_var = sum((km - km_mean) ** 2 for km in mileages) / m
    km_std = km_var**0.5
    if not math.isfinite(km_std) or km_std < EPSILON:
        raise ValueError("Mileage variance is too small for animation.")

    x_norm = [(km - km_mean) / km_std for km in mileages]

    theta0 = 0.0
    theta1 = 0.0
    frames: list[AnimationFrame] = []

    for iteration in range(1, iterations + 1):
        gradient0 = 0.0
        gradient1 = 0.0
        sq_error_sum = 0.0

        for x_value, target in zip(x_norm, prices):
            prediction = theta0 + (theta1 * x_value)
            error = prediction - target
            gradient0 += error
            gradient1 += error * x_value
            sq_error_sum += error * error

        theta0 -= learning_rate * (gradient0 / m)
        theta1 -= learning_rate * (gradient1 / m)

        raw_theta1 = theta1 / km_std
        raw_theta0 = theta0 - (theta1 * km_mean / km_std)
        mse = sq_error_sum / m

        frames.append(
            AnimationFrame(
                iteration=iteration,
                theta0=raw_theta0,
                theta1=raw_theta1,
                mse=mse,
            )
        )

    return frames
