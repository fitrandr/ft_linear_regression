from __future__ import annotations

from pathlib import Path

from evaluator.model import EPSILON

from .model import AxisTransform, PlotAnalysis
from .report import metrics_annotation


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def compute_axis_transform(mileages: list[float], x_axis: str) -> AxisTransform:
    if x_axis == "raw":
        return AxisTransform(
            x_values=list(mileages),
            x_label="Mileage (km)",
            normalized=False,
            mean=None,
            std=None,
        )

    km_mean = sum(mileages) / len(mileages)
    variance = sum((km - km_mean) ** 2 for km in mileages) / len(mileages)
    km_std = variance**0.5
    if km_std < EPSILON:
        raise ValueError("Cannot normalize x-axis: mileage variance is too small.")
    return AxisTransform(
        x_values=[(km - km_mean) / km_std for km in mileages],
        x_label="Mileage (z-score)",
        normalized=True,
        mean=km_mean,
        std=km_std,
    )


def apply_theme(theme: str):
    plt = _import_pyplot()
    if theme == "dark":
        plt.style.use("dark_background")
        return
    plt.style.use("default")


def _annotate_metrics(ax, analysis: PlotAnalysis, theme: str) -> None:
    text_color = "white" if theme == "dark" else "black"
    box_color = "#1f2937" if theme == "dark" else "#ffffff"
    ax.text(
        0.02,
        0.98,
        metrics_annotation(analysis),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=text_color,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": box_color,
            "alpha": 0.75,
            "edgecolor": "#9ca3af",
        },
    )


def render_and_save(
    mileages: list[float],
    prices: list[float],
    predictions: list[float],
    analysis: PlotAnalysis,
    output_path: Path,
    image_format: str,
    dpi: int,
    show: bool,
    theme: str,
    x_axis: str,
) -> None:
    plt = _import_pyplot()
    apply_theme(theme)

    axis_transform = compute_axis_transform(mileages, x_axis=x_axis)
    x_values = axis_transform.x_values

    order = sorted(range(len(x_values)), key=lambda idx: x_values[idx])
    x_sorted = [x_values[idx] for idx in order]
    pred_sorted = [predictions[idx] for idx in order]

    outlier_indices = [idx for idx, is_outlier in enumerate(analysis.outlier_flags) if is_outlier]
    normal_indices = [idx for idx, is_outlier in enumerate(analysis.outlier_flags) if not is_outlier]

    fig, (ax_main, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )

    if normal_indices:
        ax_main.scatter(
            [x_values[idx] for idx in normal_indices],
            [prices[idx] for idx in normal_indices],
            color="#2a9d8f",
            label="Observed points",
            s=35,
            alpha=0.85,
        )
    if outlier_indices:
        ax_main.scatter(
            [x_values[idx] for idx in outlier_indices],
            [prices[idx] for idx in outlier_indices],
            color="#e63946",
            label="Outliers",
            s=40,
            alpha=0.95,
            zorder=5,
        )

    ax_main.plot(x_sorted, pred_sorted, color="#f4a261", linewidth=2.5, label="Regression line")
    ax_main.plot(
        x_sorted,
        [analysis.baseline_value] * len(x_sorted),
        color="#64748b",
        linewidth=1.8,
        linestyle="--",
        label="Baseline mean(price)",
    )

    residual_line_colors = ["#e63946" if is_outlier else "#94a3b8" for is_outlier in analysis.outlier_flags]
    ax_main.vlines(
        x=x_values,
        ymin=predictions,
        ymax=prices,
        colors=residual_line_colors,
        alpha=0.4,
        linewidth=1.2,
        label="Residuals",
    )

    ax_main.set_title("Linear Regression Fit and Residual Diagnostics")
    ax_main.set_ylabel("Price (euros)")
    ax_main.grid(alpha=0.2)
    ax_main.legend(loc="upper right", fontsize=9)
    _annotate_metrics(ax_main, analysis, theme=theme)

    if normal_indices:
        ax_residual.scatter(
            [x_values[idx] for idx in normal_indices],
            [analysis.residuals[idx] for idx in normal_indices],
            color="#3b82f6",
            s=28,
            alpha=0.85,
            label="Residuals",
        )
    if outlier_indices:
        ax_residual.scatter(
            [x_values[idx] for idx in outlier_indices],
            [analysis.residuals[idx] for idx in outlier_indices],
            color="#ef4444",
            s=34,
            alpha=0.95,
            label="Outlier residuals",
            zorder=6,
        )

    ax_residual.vlines(
        x=x_values,
        ymin=0.0,
        ymax=analysis.residuals,
        colors=residual_line_colors,
        alpha=0.35,
        linewidth=1.0,
    )
    ax_residual.axhline(0.0, color="#9ca3af", linewidth=1.2, linestyle="-")
    ax_residual.set_xlabel(axis_transform.x_label)
    ax_residual.set_ylabel("Residual\n(actual - pred)")
    ax_residual.grid(alpha=0.2)

    if outlier_indices:
        ax_residual.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=image_format)
    if show:
        plt.show()
    plt.close(fig)
