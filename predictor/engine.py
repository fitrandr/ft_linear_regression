from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .model import (
    DEFAULT_MODEL_VERSION,
    MAX_ABS_INTERCEPT,
    MAX_ABS_PREDICTION,
    MAX_ABS_SLOPE,
    SUPPORTED_MODEL_VERSIONS,
    Model,
    ModelPolicy,
)


def _read_model_version(payload: dict[str, Any]) -> int:
    if "version" not in payload:
        return DEFAULT_MODEL_VERSION
    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError("Invalid model file: 'version' must be an integer.")
    if version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(f"Unsupported model version: {version}.")
    return version


def assert_model(model: Model) -> None:
    if not math.isfinite(model.theta0) or not math.isfinite(model.theta1):
        raise ValueError("Invalid model file: theta parameters must be finite.")
    if abs(model.theta1) > MAX_ABS_SLOPE:
        raise ValueError("Invalid model file: theta1 magnitude is unreasonable.")
    if abs(model.theta0) > MAX_ABS_INTERCEPT:
        raise ValueError("Invalid model file: theta0 magnitude is unreasonable.")


def _from_raw_thetas(model_section: dict[str, Any]) -> Model:
    if "theta0" not in model_section or "theta1" not in model_section:
        raise ValueError("Invalid model file: missing parameters 'theta0' and/or 'theta1'.")

    try:
        theta0 = float(model_section["theta0"])
        theta1 = float(model_section["theta1"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid model file: theta parameters must be numeric.") from exc

    model = Model(theta0=theta0, theta1=theta1)
    assert_model(model)
    return model


def _from_normalized_thetas(model_section: dict[str, Any]) -> Model:
    required = ("normalized_theta0", "normalized_theta1", "km_mean", "km_std")
    if not all(key in model_section for key in required):
        raise ValueError("Invalid model file: missing normalized parameters.")

    try:
        ntheta0 = float(model_section["normalized_theta0"])
        ntheta1 = float(model_section["normalized_theta1"])
        km_mean = float(model_section["km_mean"])
        km_std = float(model_section["km_std"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid model file: normalized parameters must be numeric.") from exc

    if not math.isfinite(km_std) or abs(km_std) < 1e-12:
        raise ValueError("Invalid model file: km_std must be finite and non-zero.")

    raw_theta1 = ntheta1 / km_std
    raw_theta0 = ntheta0 - (ntheta1 * km_mean / km_std)
    model = Model(theta0=raw_theta0, theta1=raw_theta1)
    assert_model(model)
    return model


def extract_model(payload: dict[str, Any]) -> Model:
    """Extract model from schema {'version': int, 'model': {...}}."""
    _read_model_version(payload)
    model_section = payload.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("Invalid model file: expected a 'model' object.")

    if "theta0" in model_section and "theta1" in model_section:
        return _from_raw_thetas(model_section)
    return _from_normalized_thetas(model_section)


def load_model(model_path: Path, model_policy: ModelPolicy, logger: logging.Logger) -> tuple[Model, bool]:
    fallback_model = Model(theta0=0.0, theta1=0.0)
    strict = model_policy == ModelPolicy.STRICT

    if not model_path.exists():
        message = f"Model file not found: {model_path}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s. Using default model (theta0=0.0, theta1=0.0).", message)
        return fallback_model, True

    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        if strict:
            raise ValueError("Invalid or corrupted model file.") from exc
        logger.warning("Invalid or corrupted model file. Using default model (theta0=0.0, theta1=0.0).")
        return fallback_model, True

    if not isinstance(payload, dict):
        if strict:
            raise ValueError("Invalid model file: expected a JSON object.")
        logger.warning("Invalid model file structure. Using default model (theta0=0.0, theta1=0.0).")
        return fallback_model, True

    try:
        model = extract_model(payload)
    except ValueError as exc:
        if strict:
            raise
        logger.warning("%s Using default model (theta0=0.0, theta1=0.0).", exc)
        return fallback_model, True

    logger.debug(
        "Loaded model from %s (theta0=%.6f, theta1=%.6f).",
        model_path,
        model.theta0,
        model.theta1,
    )
    return model, False


def predict(model: Model, mileages: list[float]) -> list[float]:
    predictions: list[float] = []
    for mileage in mileages:
        value = model.predict_single(mileage)
        if not math.isfinite(value) or abs(value) > MAX_ABS_PREDICTION:
            raise ValueError("Prediction out of bounds (model unstable).")
        predictions.append(value)
    return predictions
