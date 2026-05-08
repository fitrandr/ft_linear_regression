from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .data import load_plot_data
from .model import DEFAULT_OUTPUT_BASENAME, PlotArgs
from .render import render_and_save
from .report import build_analysis, resolve_output_path, save_report_bundle

LOGGER_NAME = "ft_linear_regression.plot"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def resolve_output_base(output_arg: str, report_dir: Path | None) -> Path:
    output_base = Path(output_arg)
    if report_dir is None:
        return output_base
    if output_base.is_absolute():
        return output_base
    if output_base.parent == Path("."):
        return report_dir / output_base.name
    return output_base


def parse_args() -> PlotArgs:
    parser = argparse.ArgumentParser(description="Plot data, regression fit, baseline and residual diagnostics.")
    parser.add_argument("--dataset", default="data.csv", help="Dataset CSV (default: data.csv)")
    parser.add_argument("--model", default="model.json", help="Model JSON (default: model.json)")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_BASENAME,
        help=f"Output image path without extension or with one (default: {DEFAULT_OUTPUT_BASENAME})",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Export format (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    parser.add_argument("--show", action="store_true", help="Display plot window after save.")
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="Plot theme (default: light)",
    )
    parser.add_argument(
        "--x-axis",
        choices=("raw", "normalized"),
        default="raw",
        help="X-axis scale (default: raw)",
    )
    parser.add_argument(
        "--report-dir",
        help="Optional directory for analysis artifacts (metrics.json, summary.txt).",
    )

    ns = parser.parse_args()

    if ns.dpi <= 0:
        parser.error("--dpi must be > 0.")

    report_dir = Path(ns.report_dir) if ns.report_dir else None

    output_base = resolve_output_base(ns.output, report_dir)

    output_path = resolve_output_path(output_base, ns.image_format)

    return PlotArgs(
        dataset_path=Path(ns.dataset),
        model_path=Path(ns.model),
        output_path=output_path,
        image_format=ns.image_format,
        show=ns.show,
        theme=ns.theme,
        x_axis=ns.x_axis,
        report_dir=report_dir,
        dpi=ns.dpi,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        loaded = load_plot_data(args.dataset_path, args.model_path, logger=logger)
        analysis = build_analysis(loaded.mileages, loaded.prices, loaded.predictions)

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        render_and_save(
            mileages=loaded.mileages,
            prices=loaded.prices,
            predictions=loaded.predictions,
            analysis=analysis,
            output_path=args.output_path,
            image_format=args.image_format,
            dpi=args.dpi,
            show=args.show,
            theme=args.theme,
            x_axis=args.x_axis,
        )

        logger.info("Plot saved to %s", args.output_path)

        if args.report_dir is not None:
            metrics_path = save_report_bundle(args.report_dir, args.output_path, analysis)
            logger.info("Metrics report saved to %s", metrics_path)

    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)
