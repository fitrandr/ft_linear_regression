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
    sample_count = len(prices)
    mean_residual_plot_space = sum(residuals) / sample_count
    residual_variance = sum((value - mean_residual_plot_space) ** 2 for value in residuals) / sample_count
    residual_std_plot_space = residual_variance**0.5

    return PlotAnalysis(
        comparison=comparison,
        correlation=correlation(mileages, prices),
        samples=sample_count,
        mileage_min=min(mileages),
        mileage_max=max(mileages),
        price_min=min(prices),
        price_max=max(prices),
        residuals=residuals,
        outlier_flags=outlier_flags,
        mean_error_model_space=mean_error,
        error_std_model_space=error_std,
        mean_residual_plot_space=mean_residual_plot_space,
        residual_std_plot_space=residual_std_plot_space,
        outlier_count=outlier_count,
        baseline_value=mean(prices),
    )


def _correlation_label(value: float | None) -> str:
    if value is None:
        return "undefined"
    strength = abs(value)
    if strength >= 0.80:
        strength_label = "very strong"
    elif strength >= 0.60:
        strength_label = "strong"
    elif strength >= 0.40:
        strength_label = "moderate"
    elif strength >= 0.20:
        strength_label = "weak"
    else:
        strength_label = "very weak"
    direction = "negative" if value < 0 else "positive"
    return f"{strength_label} {direction}"


def _usefulness_label(value: float | None) -> str:
    if value is None:
        return "undefined"
    if value >= 0.70:
        return "strong"
    if value >= 0.40:
        return "good"
    if value >= 0.20:
        return "limited"
    if value >= 0.0:
        return "weak"
    return "worse than baseline"


def metrics_annotation(analysis: PlotAnalysis) -> str:
    r2_value = (
        f"{analysis.comparison.model.r2:.4f}"
        if analysis.comparison.model.r2_defined and analysis.comparison.model.r2 is not None
        else "undefined"
    )
    correlation_value = f"{analysis.correlation:.4f}" if analysis.correlation is not None else "undefined"
    usefulness_value = analysis.comparison.usefulness_score
    usefulness_label = _usefulness_label(usefulness_value)
    snr_value = (
        f"{analysis.comparison.signal_to_noise_ratio:.3f}"
        if analysis.comparison.signal_to_noise_ratio is not None
        else "undefined"
    )
    usefulness_text = (
        f"{usefulness_value:.3f} ({usefulness_label})"
        if usefulness_value is not None
        else "undefined"
    )
    return "\n".join(
        [
            f"Samples: {analysis.samples}",
            f"Mileage range: {analysis.mileage_min:.0f} .. {analysis.mileage_max:.0f} km",
            f"Price range: {analysis.price_min:.0f} .. {analysis.price_max:.0f} EUR",
            f"R2: {r2_value}",
            f"RMSE: {analysis.comparison.model.rmse:.2f}",
            f"MAE: {analysis.comparison.model.mae:.2f}",
            f"SNR: {snr_value}",
            f"Usefulness: {usefulness_text}",
            f"Corr(km,price): {correlation_value} ({_correlation_label(analysis.correlation)})",
            f"Residual mean (actual-pred): {analysis.mean_residual_plot_space:.2f}",
            f"Residual std (actual-pred): {analysis.residual_std_plot_space:.2f}",
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
                "Plot Summary Report",
                "===================",
                f"Plot file: {plot_path}",
                f"Samples: {analysis.samples}",
                (
                    "Mileage range: "
                    f"{analysis.mileage_min:.2f} -> {analysis.mileage_max:.2f} km"
                ),
                (
                    "Price range: "
                    f"{analysis.price_min:.2f} -> {analysis.price_max:.2f} EUR"
                ),
                "",
                "Model vs baseline:",
                f"- MSE: {analysis.comparison.model.mse:.6f}",
                f"- RMSE: {analysis.comparison.model.rmse:.6f}",
                f"- MAE: {analysis.comparison.model.mae:.6f}",
                (
                    f"- R2: {analysis.comparison.model.r2:.6f}"
                    if analysis.comparison.model.r2_defined and analysis.comparison.model.r2 is not None
                    else "- R2: undefined"
                ),
                f"- Delta MSE (baseline - model): {analysis.comparison.delta_mse:.6f}",
                (
                    f"- SNR: {analysis.comparison.signal_to_noise_ratio:.6f}"
                    if analysis.comparison.signal_to_noise_ratio is not None
                    else "- SNR: undefined"
                ),
                (
                    f"- Usefulness score: {analysis.comparison.usefulness_score:.6f} "
                    f"({_usefulness_label(analysis.comparison.usefulness_score)})"
                    if analysis.comparison.usefulness_score is not None
                    else "- Usefulness score: undefined"
                ),
                "",
                "Residual diagnostics:",
                f"- Mean residual (actual - pred): {analysis.mean_residual_plot_space:.6f}",
                f"- Residual std (actual - pred): {analysis.residual_std_plot_space:.6f}",
                f"- Outliers (|z| > 3): {analysis.outlier_count}",
                (
                    f"- Correlation (km, price): {analysis.correlation:.6f} "
                    f"({_correlation_label(analysis.correlation)})"
                    if analysis.correlation is not None
                    else "- Correlation (km, price): undefined"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path
