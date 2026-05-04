from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .engine import load_model, predict
from .model import InputPolicy, ModelPolicy, PredictArgs
from .parser import parse_mileage, parse_mileage_file

LOGGER_NAME = "ft_linear_regression.predict"


def configure_logging(verbose: bool, quiet: bool) -> logging.Logger:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(message)s", force=True)
    return logging.getLogger(LOGGER_NAME)


def get_interactive_input() -> str:
    return input("Enter mileage (km): ").strip()


def _emit_output(mileages: list[float], predictions: list[float], json_output: bool) -> None:
    single = len(mileages) == 1

    if json_output:
        if single:
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

    if single:
        print(f"Estimated price: {predictions[0]:.2f} euros")
        return

    print("Estimated prices:")
    for mileage, prediction in zip(mileages, predictions):
        print(f"- mileage={mileage:.2f} km -> {prediction:.2f} euros")


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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate model and input values without running predictions.",
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

    if ns.mileage is not None and ns.mileage.strip() == "":
        parser.error("--mileage cannot be empty.")

    if ns.strict:
        model_policy = ModelPolicy.STRICT
        input_policy = InputPolicy.FAIL_FAST
    elif ns.skip_invalid:
        model_policy = ModelPolicy.NON_STRICT
        input_policy = InputPolicy.SKIP_INVALID
    else:
        model_policy = ModelPolicy.NON_STRICT
        input_policy = InputPolicy.FAIL_FAST

    return PredictArgs(
        model_path=Path(ns.model),
        json_output=ns.json_output,
        validate_only=ns.validate_only,
        verbose=ns.verbose,
        quiet=ns.quiet,
        model_policy=model_policy,
        input_policy=input_policy,
        mileage=ns.mileage,
        mileages=tuple(ns.mileages) if ns.mileages is not None else None,
        mileages_file=Path(ns.mileages_file) if ns.mileages_file is not None else None,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Validation mode must validate a real model, never a fallback.
    effective_model_policy = ModelPolicy.STRICT if args.validate_only else args.model_policy

    try:
        model, used_fallback = load_model(args.model_path, model_policy=effective_model_policy, logger=logger)
        if args.mileage is not None:
            raw_values = [args.mileage]
        elif args.mileages is not None:
            raw_values = list(args.mileages)
        elif args.mileages_file is not None:
            raw_values = parse_mileage_file(args.mileages_file)
        else:
            raw_values = [get_interactive_input()]
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    if args.validate_only and used_fallback:
        logger.error("Error: validate-only does not allow fallback model.")
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

    if args.validate_only:
        if args.json_output:
            print(json.dumps({"valid": True, "count": len(mileages)}, indent=2))
        else:
            print(f"Validation successful: {len(mileages)} mileage value(s).")
        return

    try:
        predictions = predict(model, mileages)
    except ValueError as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    _emit_output(mileages, predictions, json_output=args.json_output)
