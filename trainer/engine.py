from __future__ import annotations

import logging

from .data import validate_pairs
from .model import EPSILON, LinearModel, TrainingLogEntry, TrainingResult


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return sum(values) / len(values)


def std(values: list[float], values_mean: float) -> float:
    if not values:
        raise ValueError("Cannot compute std of an empty list.")
    variance = sum((value - values_mean) ** 2 for value in values) / len(values)
    return variance**0.5


def mse(mileages: list[float], prices: list[float], model: LinearModel) -> float:
    validate_pairs(mileages, prices)
    m = len(mileages)
    total = 0.0
    for km, price in zip(mileages, prices):
        error = model.predict(km) - price
        total += error * error
    return total / m


def train_model(
    mileages: list[float],
    prices: list[float],
    learning_rate: float,
    iterations: int,
    log_every: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    logger: logging.Logger,
) -> TrainingResult:
    validate_pairs(mileages, prices)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")
    if log_every < 0:
        raise ValueError("log_every must be >= 0.")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be > 0.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")

    m = len(mileages)
    km_mean = mean(mileages)
    km_std = std(mileages, km_mean)
    if km_std < EPSILON:
        raise ValueError("Mileage variance is too small; cannot train a stable slope.")

    norm_mileages = [(km - km_mean) / km_std for km in mileages]
    theta0 = 0.0
    theta1 = 0.0
    history: list[TrainingLogEntry] = []
    best_train_mse = float("inf")
    no_improvement_count = 0
    stopped_early = False
    iterations_ran = iterations

    for iteration in range(1, iterations + 1):
        gradient0 = 0.0
        gradient1 = 0.0
        squared_error_sum = 0.0

        for mileage, price in zip(norm_mileages, prices):
            prediction = theta0 + theta1 * mileage
            error = prediction - price
            gradient0 += error
            gradient1 += error * mileage
            squared_error_sum += error * error

        should_log = (
            log_every > 0
            and (iteration == 1 or iteration == iterations or iteration % log_every == 0)
        )
        train_mse = squared_error_sum / m

        if should_log:
            history.append(TrainingLogEntry(iteration=iteration, mse=train_mse))
            logger.info("Iteration %d/%d - Train MSE: %.6f", iteration, iterations, train_mse)

        theta0 -= learning_rate * (gradient0 / m)
        theta1 -= learning_rate * (gradient1 / m)

        if train_mse < best_train_mse - early_stopping_min_delta:
            best_train_mse = train_mse
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            stopped_early = True
            iterations_ran = iteration
            if log_every > 0 and not should_log:
                history.append(TrainingLogEntry(iteration=iteration, mse=train_mse))
            logger.info(
                "Early stopping at iteration %d (best train MSE: %.6f).",
                iteration,
                best_train_mse,
            )
            break

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
        iterations_ran=iterations_ran,
        log_every=log_every,
        samples=m,
        stopped_early=stopped_early,
        best_train_mse=best_train_mse,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        history=tuple(history),
    )
