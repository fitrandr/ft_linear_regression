from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .engine import build_interpretation_text, load_interpreted_report, save_interpretation
from .model import InterpretArgs, LOGGER_NAME


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def parse_args() -> InterpretArgs:
    parser = argparse.ArgumentParser(
        description="Interpret an evaluation report and save a human-readable summary."
    )
    parser.add_argument(
        "--report",
        default="evaluation_report.json",
        help="Path to evaluation JSON report (default: evaluation_report.json)",
    )
    parser.add_argument(
        "--output",
        default="interpretation_report.txt",
        help="Path to output interpretation file (default: interpretation_report.txt)",
    )
    parser.add_argument(
        "--print",
        dest="print_output",
        action="store_true",
        help="Also print the interpretation in terminal.",
    )
    ns = parser.parse_args()
    return InterpretArgs(
        report_path=Path(ns.report),
        output_path=Path(ns.output),
        print_output=ns.print_output,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        report = load_interpreted_report(args.report_path)
        content = build_interpretation_text(report)
        save_interpretation(args.output_path, content)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)

    logger.info("Interpretation saved to %s", args.output_path)
    if args.print_output:
        print(content)
