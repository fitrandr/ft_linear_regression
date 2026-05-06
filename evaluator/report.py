from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .model import EvaluationResult, MetricsComparison


def _format_r2(value: float | None, defined: bool) -> str:
    return "undefined (constant target values)" if not defined else f"{value:.6f}"


def _format_optional(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def serialize_result(result: EvaluationResult) -> dict[str, object]:
    return asdict(result)


def render_text_output(result: EvaluationResult) -> str:
    lines: list[str] = [
        f"Samples      : {result.samples}",
        (
            "Split        : "
            f"{result.split.train_samples} train / {result.split.test_samples} test "
            f"(ratio={result.split.test_ratio}, seed={result.split.seed})"
        ),
    ]

    def append_comparison(title: str, comparison: MetricsComparison) -> None:
        lines.append(title)
        lines.append(f"MAE          : {comparison.model.mae:.6f}")
        lines.append(f"MSE          : {comparison.model.mse:.6f}")
        lines.append(f"RMSE         : {comparison.model.rmse:.6f}")
        lines.append(f"R2           : {_format_r2(comparison.model.r2, comparison.model.r2_defined)}")
        lines.append(f"Mean Error   : {comparison.model.mean_error:.6f}")
        lines.append(f"Error Std    : {comparison.model.error_std:.6f}")
        lines.append(f"Max Abs Error: {comparison.model.max_abs_error:.6f}")
        lines.append(f"Outliers     : {comparison.model.outlier_count}")
        lines.append("Baseline metrics (predict mean(price))")
        lines.append(f"MAE          : {comparison.baseline.mae:.6f}")
        lines.append(f"MSE          : {comparison.baseline.mse:.6f}")
        lines.append(f"RMSE         : {comparison.baseline.rmse:.6f}")
        lines.append(
            "R2           : "
            f"{_format_r2(comparison.baseline.r2, comparison.baseline.r2_defined)}"
        )
        lines.append(f"Mean Error   : {comparison.baseline.mean_error:.6f}")
        lines.append(f"Error Std    : {comparison.baseline.error_std:.6f}")
        lines.append(f"Max Abs Error: {comparison.baseline.max_abs_error:.6f}")
        lines.append(f"Outliers     : {comparison.baseline.outlier_count}")
        lines.append(f"Delta MSE    : {comparison.delta_mse:.6f} (positive means model is better)")
        lines.append(f"SNR          : {_format_optional(comparison.signal_to_noise_ratio)}")
        lines.append(f"Usefulness   : {_format_optional(comparison.usefulness_score)}")

    append_comparison("Full dataset metrics", result.full)
    append_comparison("Train split metrics", result.train)
    append_comparison("Test split metrics", result.test)
    if result.mileage_price_correlation is None:
        lines.append("Correlation  : undefined")
    else:
        lines.append(f"Correlation  : {result.mileage_price_correlation:.6f}")

    return "\n".join(lines)


def emit_output(result: EvaluationResult, json_output: bool) -> None:
    if json_output:
        print(json.dumps(serialize_result(result), indent=2))
        return
    print(render_text_output(result))


def save_report(path: Path, result: EvaluationResult) -> None:
    path.write_text(json.dumps(serialize_result(result), indent=2), encoding="utf-8")
