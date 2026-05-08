from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .diagnostics import build_gradient_descent_frames
from .data import load_plot_data
from .export import build_report_image_paths, default_animation_path
from .model import DEFAULT_OUTPUT_BASENAME, PlotArgs
from .render import render_and_save, render_report_images, render_training_animation
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
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio used for train/test visualization (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed used for train/test visualization (default: 42).",
    )
    parser.add_argument(
        "--generate-report-images",
        action="store_true",
        help=(
            "When --report-dir is set, also generate regression/residuals/"
            "actual-vs-predicted/error-distribution images."
        ),
    )
    parser.add_argument(
        "--animate-training",
        action="store_true",
        help="Generate an educational GIF animation of gradient descent convergence.",
    )
    parser.add_argument(
        "--animation-iterations",
        type=int,
        default=120,
        help="Number of gradient descent steps used to render animation (default: 120).",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=8,
        help="Animation FPS (default: 8).",
    )

    ns = parser.parse_args()

    if ns.dpi <= 0:
        parser.error("--dpi must be > 0.")
    if not 0.0 < ns.test_ratio < 1.0:
        parser.error("--test-ratio must be in the interval (0, 1).")
    if ns.animation_iterations <= 0:
        parser.error("--animation-iterations must be > 0.")
    if ns.animation_fps <= 0:
        parser.error("--animation-fps must be > 0.")

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
        test_ratio=ns.test_ratio,
        seed=ns.seed,
        report_dir=report_dir,
        generate_report_images=ns.generate_report_images,
        animate_training=ns.animate_training,
        animation_iterations=ns.animation_iterations,
        animation_fps=ns.animation_fps,
        dpi=ns.dpi,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging()

    try:
        loaded = load_plot_data(
            args.dataset_path,
            args.model_path,
            logger=logger,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        analysis = build_analysis(loaded.mileages, loaded.prices, loaded.predictions, loaded.model)

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        render_and_save(
            mileages=loaded.mileages,
            prices=loaded.prices,
            predictions=loaded.predictions,
            is_test_flags=loaded.is_test_flags,
            model=loaded.model,
            analysis=analysis,
            output_path=args.output_path,
            image_format=args.image_format,
            dpi=args.dpi,
            show=args.show,
            theme_name=args.theme,
            x_axis=args.x_axis,
        )

        logger.info("Dashboard plot saved to %s", args.output_path)

        if args.report_dir is not None:
            metrics_path = save_report_bundle(args.report_dir, args.output_path, analysis)
            logger.info("Metrics report saved to %s", metrics_path)

            if args.generate_report_images:
                image_paths = build_report_image_paths(args.report_dir, args.image_format)
                render_report_images(
                    mileages=loaded.mileages,
                    prices=loaded.prices,
                    predictions=loaded.predictions,
                    is_test_flags=loaded.is_test_flags,
                    analysis=analysis,
                    theme_name=args.theme,
                    x_axis=args.x_axis,
                    image_paths=image_paths,
                    dpi=args.dpi,
                )
                logger.info(
                    "Report images saved to %s, %s, %s, %s",
                    image_paths.regression,
                    image_paths.residuals,
                    image_paths.predicted_vs_actual,
                    image_paths.error_distribution,
                )

        if args.animate_training:
            animation_path = default_animation_path(args.output_path)
            frames = build_gradient_descent_frames(
                loaded.mileages,
                loaded.prices,
                iterations=args.animation_iterations,
                learning_rate=0.1,
            )
            render_training_animation(
                mileages=loaded.mileages,
                prices=loaded.prices,
                frames=frames,
                output_path=animation_path,
                dpi=args.dpi,
                fps=args.animation_fps,
                theme_name=args.theme,
                is_test_flags=loaded.is_test_flags,
            )
            logger.info("Training animation saved to %s", animation_path)

    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.error("Error: %s", exc)
        raise SystemExit(1)
