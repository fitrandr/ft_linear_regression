from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from evaluator.stats import compare_with_baseline, compute_errors, compute_variance_stats, correlation, mean

from .model import PlotAnalysis


def resolve_output_path(base_path: Path, image_format: str) -> Path:
    suffix = f".{image_format.lower()}"
    if base_path.suffix.lower() == suffix:
        return base_path
    if base_path.suffix:
        return base_path.with_suffix(suffix)
    return base_path.with_name(base_path.name + suffix)


def build_analysis(mileages: list[float], prices: list[float], predictions: list[float]) -> PlotAnalysis:
    comparison = compare_with_baseline(prices, predictions)
    residual_stats = compute_errors(prices, predictions)
    mean_error, error_std, outlier_count = compute_variance_stats(
        residual_stats.residuals,
        residual_stats.sq_error_sum,
    )

    outlier_flags: list[bool] = []
    if error_std > 0.0:
        for residual in residual_stats.residuals:
            z_score = (residual - mean_error) / error_std
            outlier_flags.append(abs(z_score) > 3.0)
    else:
        outlier_flags = [False] * len(residual_stats.residuals)

    # Plot convention: residual = actual - predicted.
    residuals = [actual - predicted for actual, predicted in zip(prices, predictions)]

    return PlotAnalysis(
        comparison=comparison,
        correlation=correlation(mileages, prices),
        residuals=residuals,
        outlier_flags=outlier_flags,
        mean_error_model_space=mean_error,
        error_std_model_space=error_std,
        outlier_count=outlier_count,
        baseline_value=mean(prices),
    )


def metrics_annotation(analysis: PlotAnalysis) -> str:
    r2_value = (
        f"{analysis.comparison.model.r2:.4f}"
        if analysis.comparison.model.r2_defined and analysis.comparison.model.r2 is not None
        else "undefined"
    )
    correlation_value = (
        f"{analysis.correlation:.4f}" if analysis.correlation is not None else "undefined"
    )
    return "\n".join(
        [
            f"R2: {r2_value}",
            f"RMSE: {analysis.comparison.model.rmse:.2f}",
            f"MAE: {analysis.comparison.model.mae:.2f}",
            f"SNR: {analysis.comparison.signal_to_noise_ratio:.3f}"
            if analysis.comparison.signal_to_noise_ratio is not None
            else "SNR: undefined",
            f"Usefulness: {analysis.comparison.usefulness_score:.3f}"
            if analysis.comparison.usefulness_score is not None
            else "Usefulness: undefined",
            f"Corr(km,price): {correlation_value}",
            f"Outliers: {analysis.outlier_count}",
        ]
    )


def serialize_analysis(analysis: PlotAnalysis) -> dict[str, object]:
    return asdict(analysis)


def save_metrics_json(path: Path, analysis: PlotAnalysis) -> None:
    path.write_text(json.dumps(serialize_analysis(analysis), indent=2), encoding="utf-8")


def save_report_bundle(report_dir: Path, plot_path: Path, analysis: PlotAnalysis) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    save_metrics_json(metrics_path, analysis)

    summary_path = report_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"plot_file={plot_path}",
                f"mse={analysis.comparison.model.mse:.6f}",
                f"rmse={analysis.comparison.model.rmse:.6f}",
                f"mae={analysis.comparison.model.mae:.6f}",
                f"outliers={analysis.outlier_count}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path
