#!/usr/bin/env python3
"""CLI tool for predicting car prices using trained linear regression model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LOGGER_NAME = "ft_linear_regression.predict"
MAX_ABS_SLOPE = 1_000_000.0
MAX_ABS_INTERCEPT = 1_000_000_000.0
THOUSANDS_GROUP_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

InvalidPolicy = Literal["strict", "skip-invalid", "fail-fast"]


@dataclass(frozen=True)
class Model:
    theta0: float
    theta1: float

    def predict(self, mileage: float) -> float:
        return self.theta0 + self.theta1 * mileage


@dataclass(frozen=True)
class PredictArgs:
    model_path: Path
    json_output: bool
    verbose: bool
    quiet: bool
    invalid_policy: InvalidPolicy
    mileage: str | None
    mileages: tuple[str, ...] | None
    mileages_file: Path | None

    @property
    def strict_model(self) -> bool:
        return self.invalid_policy == "strict"

    @property
    def fail_fast(self) -> bool:
        return self.invalid_policy in ("strict", "fail-fast")


def configure_logging(verbose: bool, quiet: bool) -> logging.Logger:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(message)s", force=True)
    return logging.getLogger(LOGGER_NAME)


def extract_model(model_data: dict[str, Any]) -> Model:
    """Extract and validate model from strict schema: {'model': {'theta0', 'theta1'}}."""
    model_section = model_data.get("model")
    if not isinstance(model_section, dict):
        raise ValueError("Invalid model file: expected a 'model' object.")

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


def assert_model(model: Model) -> None:
    """Strong model validation for production safety."""
    if not math.isfinite(model.theta0) or not math.isfinite(model.theta1):
        raise ValueError("Invalid model file: theta parameters must be finite.")
    if abs(model.theta1) > MAX_ABS_SLOPE:
        raise ValueError("Invalid model file: theta1 magnitude is unreasonable.")
    if abs(model.theta0) > MAX_ABS_INTERCEPT:
        raise ValueError("Invalid model file: theta0 magnitude is unreasonable.")


def load_model(model_path: Path, strict_model: bool, logger: logging.Logger) -> Model:
    """Load model. In non-strict mode, fallback to zero model with explicit warning."""
    fallback_model = Model(theta0=0.0, theta1=0.0)

    if not model_path.exists():
        message = f"Model file not found: {model_path}"
        if strict_model:
            raise FileNotFoundError(message)
        logger.warning("%s. Using default model (theta0=0.0, theta1=0.0).", message)
        return fallback_model

    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        if strict_model:
            raise ValueError("Invalid or corrupted model file.") from exc
        logger.warning("Invalid or corrupted model file. Using default model (theta0=0.0, theta1=0.0).")
        return fallback_model

    if not isinstance(payload, dict):
        if strict_model:
            raise ValueError("Invalid model file: expected a JSON object.")
        logger.warning("Invalid model file structure. Using default model (theta0=0.0, theta1=0.0).")
        return fallback_model

    try:
        model = extract_model(payload)
    except ValueError as exc:
        if strict_model:
            raise
        logger.warning("%s Using default model (theta0=0.0, theta1=0.0).", exc)
        return fallback_model

    logger.debug(
        "Loaded model from %s (theta0=%.6f, theta1=%.6f).",
        model_path,
        model.theta0,
        model.theta1,
    )
    return model


def normalize_number(value: str) -> str:
    """Normalize numeric string by removing spaces/underscores and thousands commas."""
    normalized = value.strip()
    if normalized == "":
        raise ValueError("Mileage cannot be empty.")

    normalized = normalized.replace(" ", "").replace("_", "")

    if "," in normalized:
        if THOUSANDS_GROUP_RE.fullmatch(normalized):
            normalized = normalized.replace(",", "")
        else:
            raise ValueError(f"Invalid mileage format: {value!r}")

    return normalized


def parse_float(value: str) -> float:
    """Parse finite float from normalized numeric string."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError("Value must be finite.")
    return parsed


def parse_mileage(raw_value: str) -> float:
    normalized = normalize_number(raw_value)
    mileage = parse_float(normalized)
    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    return mileage


def parse_mileage_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Mileage input file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    values = [line.strip() for line in lines if line.strip()]
    if not values:
        raise ValueError("Mileage input file is empty.")
    return values


def get_interactive_input() -> str:
    return input("Enter mileage (km): ").strip()


def predict_batch(model: Model, mileages: list[float]) -> list[float]:
    return [model.predict(mileage) for mileage in mileages]


def parse_args() -> PredictArgs:
    parser = argparse.ArgumentParser(description="Predict car price for one or many mileage values.")
    parser.add_argument(
        "--model",
        default="model.json",
        help="Path to model file (default: model.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output predictions as JSON.",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--mileage",
        help="Single mileage value. If omitted, interactive input is used.",
    )
    input_group.add_argument(
        "--mileages",
        nargs="+",
        help="Batch mileage values, e.g. --mileages 10000 20000 30000",
    )
    input_group.add_argument(
        "--mileages-file",
        help="Path to a file containing mileage values (one per line).",
    )

    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: strict model loading + fail fast on invalid mileage.",
    )
    policy_group.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid mileage values and continue processing others.",
    )
    policy_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail immediately on first invalid mileage value.",
    )

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logs.",
    )
    verbosity_group.add_argument(
        "--quiet",
        action="store_true",
        help="Show only errors.",
    )

    ns = parser.parse_args()

    if ns.strict:
        invalid_policy: InvalidPolicy = "strict"
    elif ns.skip_invalid:
        invalid_policy = "skip-invalid"
    elif ns.fail_fast:
        invalid_policy = "fail-fast"
    else:
        invalid_policy = "fail-fast"

    return PredictArgs(
        model_path=Path(ns.model),
        json_output=ns.json_output,
        verbose=ns.verbose,
        quiet=ns.quiet,
        invalid_policy=invalid_policy,
        mileage=ns.mileage,
        mileages=tuple(ns.mileages) if ns.mileages is not None else None,
        mileages_file=Path(ns.mileages_file) if ns.mileages_file is not None else None,
    )


def collect_raw_mileages(args: PredictArgs) -> list[str]:
    if args.mileage is not None:
        return [args.mileage]
    if args.mileages is not None:
        return list(args.mileages)
    if args.mileages_file is not None:
        return parse_mileage_file(args.mileages_file)
    return [get_interactive_input()]


def emit_output(mileages: list[float], predictions: list[float], json_output: bool) -> None:
    if json_output:
        if len(mileages) == 1:
            print(json.dumps({"mileage": mileages[0], "prediction": predictions[0]}, indent=2))
            return
        payload = {
            "predictions": [
                {"mileage": mileage, "prediction": prediction}
                for mileage, prediction in zip(mileages, predictions)
            ]
        }
        print(json.dumps(payload, indent=2))
        return

    if len(mileages) == 1:
        print(f"Estimated price: {predictions[0]:.2f} euros")
        return

    print("Estimated prices:")
    for mileage, prediction in zip(mileages, predictions):
        print(f"- mileage={mileage:.2f} km -> {prediction:.2f} euros")


def main() -> None:
    args = parse_args()
    logger = configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        model = load_model(args.model_path, strict_model=args.strict_model, logger=logger)
        raw_values = collect_raw_mileages(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    mileages: list[float] = []
    for raw_value in raw_values:
        try:
            mileages.append(parse_mileage(raw_value))
        except ValueError as exc:
            if args.fail_fast:
                logger.error("Error: %s", exc)
                raise SystemExit(1)
            logger.warning("Skipping invalid mileage %r: %s", raw_value, exc)

    if not mileages:
        logger.error("No valid mileage values to predict.")
        raise SystemExit(1)

    predictions = predict_batch(model, mileages)
    emit_output(mileages, predictions, json_output=args.json_output)


if __name__ == "__main__":
    main()
