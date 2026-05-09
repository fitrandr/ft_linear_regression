from __future__ import annotations

import os
from pathlib import Path

from evaluator.model import EPSILON
from predictor.model import Model

from .diagnostics import AnimationFrame
from .export import ReportImagePaths
from .model import AxisTransform, PlotAnalysis
from .report import metrics_annotation
from .theme import PlotTheme, apply_theme_style, resolve_theme


def _resolve_output_color(theme: PlotTheme, output_color: str | None) -> str:
    return theme.regression if output_color is None else output_color


def _import_pyplot(show: bool):
    try:
        if "MPLCONFIGDIR" not in os.environ:
            os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
            os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

        import matplotlib

        if not show:
            matplotlib.use("Agg", force=True)

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


def _annotate_metrics(ax, analysis: PlotAnalysis, model: Model, theme: PlotTheme) -> None:
    ax.text(
        0.02,
        0.98,
        metrics_annotation(analysis, model),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        color=theme.text,
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": theme.axes_background,
            "alpha": 0.80,
            "edgecolor": theme.grid,
        },
    )


def _annotate_residuals(ax, analysis: PlotAnalysis, theme: PlotTheme) -> None:
    threshold = 3.0 * analysis.residual_std_plot_space
    text = "\n".join(
        [
            f"Residual mean (pred-actual): {analysis.mean_residual_plot_space:.2f}",
            f"Residual std (pred-actual): {analysis.residual_std_plot_space:.2f}",
            f"Outlier threshold (|z|>3): {threshold:.2f}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        color=theme.text,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": theme.axes_background,
            "alpha": 0.78,
            "edgecolor": theme.grid,
        },
    )


def _annotate_top_residuals(ax, x_values: list[float], residuals: list[float], top_k: int = 3) -> None:
    if not residuals:
        return
    ranked_indices = sorted(range(len(residuals)), key=lambda idx: abs(residuals[idx]), reverse=True)
    for idx in ranked_indices[:top_k]:
        ax.annotate(
            f"{residuals[idx]:.0f}",
            (x_values[idx], residuals[idx]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            alpha=0.9,
        )


def _indices(is_test_flags: list[bool], outlier_flags: list[bool]) -> dict[str, list[int]]:
    groups = {
        "train_normal": [],
        "test_normal": [],
        "train_outlier": [],
        "test_outlier": [],
    }
    for idx, (is_test, is_outlier) in enumerate(zip(is_test_flags, outlier_flags)):
        if is_test and is_outlier:
            groups["test_outlier"].append(idx)
        elif is_test and not is_outlier:
            groups["test_normal"].append(idx)
        elif not is_test and is_outlier:
            groups["train_outlier"].append(idx)
        else:
            groups["train_normal"].append(idx)
    return groups


def _plot_regression_panel(
    ax,
    x_values: list[float],
    prices: list[float],
    predictions: list[float],
    analysis: PlotAnalysis,
    groups: dict[str, list[int]],
    theme: PlotTheme,
    output_color: str | None,
) -> None:
    accent_color = _resolve_output_color(theme, output_color)
    order = sorted(range(len(x_values)), key=lambda idx: x_values[idx])
    x_sorted = [x_values[idx] for idx in order]
    pred_sorted = [predictions[idx] for idx in order]

    if groups["train_normal"]:
        ax.scatter(
            [x_values[idx] for idx in groups["train_normal"]],
            [prices[idx] for idx in groups["train_normal"]],
            color=theme.train,
            label="Train points",
            s=28,
            alpha=0.85,
        )
    if groups["test_normal"]:
        ax.scatter(
            [x_values[idx] for idx in groups["test_normal"]],
            [prices[idx] for idx in groups["test_normal"]],
            color=theme.test,
            label="Test points",
            s=34,
            alpha=0.92,
        )

    if groups["train_outlier"]:
        ax.scatter(
            [x_values[idx] for idx in groups["train_outlier"]],
            [prices[idx] for idx in groups["train_outlier"]],
            color=theme.outlier,
            label="Train outliers",
            marker="x",
            s=55,
            alpha=0.95,
            zorder=6,
        )
    if groups["test_outlier"]:
        ax.scatter(
            [x_values[idx] for idx in groups["test_outlier"]],
            [prices[idx] for idx in groups["test_outlier"]],
            color=theme.outlier,
            label="Test outliers",
            marker="D",
            s=48,
            alpha=0.95,
            zorder=6,
        )

    ax.plot(x_sorted, pred_sorted, color=accent_color, linewidth=2.4, label="Regression line")
    ax.plot(
        x_sorted,
        [analysis.baseline_value] * len(x_sorted),
        color=theme.baseline,
        linewidth=1.7,
        linestyle="--",
        label="Baseline mean(price)",
    )

    residual_line_colors = [theme.outlier if flag else theme.residual_line for flag in analysis.outlier_flags]
    ax.vlines(
        x=x_values,
        ymin=predictions,
        ymax=prices,
        colors=residual_line_colors,
        alpha=0.35,
        linewidth=1.1,
        label="Residuals",
    )

    ax.set_title("Regression Fit (Train/Test + Outliers)")
    ax.set_ylabel("Price (euros)")
    ax.grid(alpha=0.25, color=theme.grid)
    ax.legend(loc="best", fontsize=8)


def _plot_residual_panel(
    ax,
    x_values: list[float],
    residuals: list[float],
    groups: dict[str, list[int]],
    analysis: PlotAnalysis,
    theme: PlotTheme,
) -> None:
    if groups["train_normal"]:
        ax.scatter(
            [x_values[idx] for idx in groups["train_normal"]],
            [residuals[idx] for idx in groups["train_normal"]],
            color=theme.train,
            s=24,
            alpha=0.85,
            label="Train residuals",
        )
    if groups["test_normal"]:
        ax.scatter(
            [x_values[idx] for idx in groups["test_normal"]],
            [residuals[idx] for idx in groups["test_normal"]],
            color=theme.test,
            s=30,
            alpha=0.92,
            label="Test residuals",
        )

    outlier_indices = groups["train_outlier"] + groups["test_outlier"]
    if outlier_indices:
        ax.scatter(
            [x_values[idx] for idx in outlier_indices],
            [residuals[idx] for idx in outlier_indices],
            color=theme.outlier,
            s=36,
            alpha=0.95,
            label="Outlier residuals",
            zorder=6,
        )

    residual_line_colors = [theme.outlier if flag else theme.residual_line for flag in analysis.outlier_flags]
    ax.vlines(
        x=x_values,
        ymin=0.0,
        ymax=residuals,
        colors=residual_line_colors,
        alpha=0.35,
        linewidth=1.0,
    )
    ax.axhline(0.0, color=theme.grid, linewidth=1.2)
    ax.set_title("Residual Plot (prediction - actual)")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25, color=theme.grid)
    ax.legend(loc="best", fontsize=8)


def _plot_error_histogram(
    ax,
    residuals: list[float],
    theme: PlotTheme,
    output_color: str | None,
) -> None:
    accent_color = _resolve_output_color(theme, output_color)
    ax.hist(residuals, bins=15, color=accent_color, alpha=0.75, edgecolor=theme.grid)
    mean_residual = sum(residuals) / len(residuals)
    ax.axvline(0.0, color=theme.grid, linewidth=1.3, linestyle="--", label="Zero error")
    ax.axvline(mean_residual, color=theme.outlier, linewidth=1.3, label="Mean residual")
    ax.set_title("Error Distribution")
    ax.set_xlabel("Residual (prediction - actual)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.20, color=theme.grid)
    ax.legend(loc="best", fontsize=8)


def _plot_actual_vs_predicted(
    ax,
    prices: list[float],
    predictions: list[float],
    groups: dict[str, list[int]],
    theme: PlotTheme,
    analysis: PlotAnalysis,
) -> None:
    r2_text = (
        f"{analysis.comparison.model.r2:.3f}"
        if analysis.comparison.model.r2_defined and analysis.comparison.model.r2 is not None
        else "undefined"
    )

    if groups["train_normal"]:
        ax.scatter(
            [prices[idx] for idx in groups["train_normal"]],
            [predictions[idx] for idx in groups["train_normal"]],
            color=theme.train,
            s=26,
            alpha=0.85,
            label="Train",
        )
    if groups["test_normal"]:
        ax.scatter(
            [prices[idx] for idx in groups["test_normal"]],
            [predictions[idx] for idx in groups["test_normal"]],
            color=theme.test,
            s=32,
            alpha=0.92,
            label="Test",
        )

    outlier_indices = groups["train_outlier"] + groups["test_outlier"]
    if outlier_indices:
        ax.scatter(
            [prices[idx] for idx in outlier_indices],
            [predictions[idx] for idx in outlier_indices],
            color=theme.outlier,
            s=38,
            alpha=0.96,
            label="Outliers",
            zorder=6,
        )

    min_value = min(min(prices), min(predictions))
    max_value = max(max(prices), max(predictions))
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color=theme.baseline,
        linestyle="--",
        linewidth=1.5,
        label="Ideal diagonal",
    )

    ax.text(
        0.02,
        0.98,
        (
            f"RMSE={analysis.comparison.model.rmse:.2f}\n"
            f"R2={r2_text}\n"
            f"Quality={analysis.quality_label.upper()}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.3,
        color=theme.text,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": theme.axes_background,
            "alpha": 0.78,
            "edgecolor": theme.grid,
        },
    )

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.grid(alpha=0.25, color=theme.grid)
    ax.legend(loc="best", fontsize=8)


def _draw_dashboard(
    plt,
    mileages: list[float],
    prices: list[float],
    predictions: list[float],
    is_test_flags: list[bool],
    model: Model,
    analysis: PlotAnalysis,
    theme: PlotTheme,
    output_color: str | None,
    x_axis: str,
):
    axis_transform = compute_axis_transform(mileages, x_axis=x_axis)
    x_values = axis_transform.x_values

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_main = axes[0][0]
    ax_residual = axes[0][1]
    ax_hist = axes[1][0]
    ax_pred = axes[1][1]

    groups = _indices(is_test_flags, analysis.outlier_flags)

    _plot_regression_panel(
        ax_main,
        x_values,
        prices,
        predictions,
        analysis,
        groups,
        theme,
        output_color,
    )
    _plot_residual_panel(ax_residual, x_values, analysis.residuals, groups, analysis, theme)
    _plot_error_histogram(ax_hist, analysis.residuals, theme, output_color)
    _plot_actual_vs_predicted(ax_pred, prices, predictions, groups, theme, analysis)

    _annotate_metrics(ax_main, analysis, model, theme)
    _annotate_residuals(ax_residual, analysis, theme)
    _annotate_top_residuals(ax_residual, x_values, analysis.residuals)

    ax_main.set_xlabel(axis_transform.x_label)
    ax_residual.set_xlabel(axis_transform.x_label)

    if axis_transform.normalized and axis_transform.mean is not None and axis_transform.std is not None:
        fig.suptitle(
            (
                "Linear Regression Analytics Dashboard "
                f"(normalized x-axis; mean={axis_transform.mean:.2f}, std={axis_transform.std:.2f})"
            ),
            fontsize=13,
        )
    else:
        fig.suptitle("Linear Regression Analytics Dashboard (raw mileage axis)", fontsize=13)

    return fig, axis_transform, groups


def _save_single_panel(
    plt,
    theme: PlotTheme,
    fig_size: tuple[float, float],
    output_path: Path,
    dpi: int,
    draw_fn,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    draw_fn(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, format=output_path.suffix.lstrip("."))
    plt.close(fig)


def render_report_images(
    mileages: list[float],
    prices: list[float],
    predictions: list[float],
    is_test_flags: list[bool],
    analysis: PlotAnalysis,
    theme_name: str,
    output_color: str | None,
    x_axis: str,
    image_paths: ReportImagePaths,
    dpi: int,
) -> None:
    plt = _import_pyplot(show=False)
    theme = resolve_theme(theme_name)
    apply_theme_style(plt, theme)

    axis_transform = compute_axis_transform(mileages, x_axis=x_axis)
    x_values = axis_transform.x_values
    groups = _indices(is_test_flags, analysis.outlier_flags)

    def draw_regression(ax):
        _plot_regression_panel(
            ax,
            x_values,
            prices,
            predictions,
            analysis,
            groups,
            theme,
            output_color,
        )
        ax.set_xlabel(axis_transform.x_label)

    def draw_residual(ax):
        _plot_residual_panel(ax, x_values, analysis.residuals, groups, analysis, theme)
        ax.set_xlabel(axis_transform.x_label)
        _annotate_residuals(ax, analysis, theme)

    def draw_hist(ax):
        _plot_error_histogram(ax, analysis.residuals, theme, output_color)

    def draw_actual_vs_pred(ax):
        _plot_actual_vs_predicted(ax, prices, predictions, groups, theme, analysis)

    _save_single_panel(plt, theme, (11, 6), image_paths.regression, dpi, draw_regression)
    _save_single_panel(plt, theme, (11, 6), image_paths.residuals, dpi, draw_residual)
    _save_single_panel(plt, theme, (10, 6), image_paths.error_distribution, dpi, draw_hist)
    _save_single_panel(plt, theme, (10, 6), image_paths.predicted_vs_actual, dpi, draw_actual_vs_pred)


def render_training_animation(
    mileages: list[float],
    prices: list[float],
    frames: list[AnimationFrame],
    output_path: Path,
    dpi: int,
    fps: int,
    theme_name: str,
    output_color: str | None,
    is_test_flags: list[bool],
) -> None:
    if not frames:
        raise ValueError("Animation frames list is empty.")

    plt = _import_pyplot(show=False)
    theme = resolve_theme(theme_name)
    apply_theme_style(plt, theme)

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise ValueError("Animation export requires matplotlib animation + pillow support.") from exc

    x_min = min(mileages)
    x_max = max(mileages)
    x_line = [x_min + (x_max - x_min) * step / 100.0 for step in range(101)]

    fig, ax = plt.subplots(figsize=(10, 6))

    train_indices = [idx for idx, is_test in enumerate(is_test_flags) if not is_test]
    test_indices = [idx for idx, is_test in enumerate(is_test_flags) if is_test]

    if train_indices:
        ax.scatter(
            [mileages[idx] for idx in train_indices],
            [prices[idx] for idx in train_indices],
            color=theme.train,
            alpha=0.8,
            s=26,
            label="Train points",
        )
    if test_indices:
        ax.scatter(
            [mileages[idx] for idx in test_indices],
            [prices[idx] for idx in test_indices],
            color=theme.test,
            alpha=0.92,
            s=30,
            label="Test points",
        )

    line, = ax.plot(
        [],
        [],
        color=_resolve_output_color(theme, output_color),
        linewidth=2.5,
        label="Gradient descent line",
    )
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax.set_title("Gradient Descent Animation")
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25, color=theme.grid)
    ax.legend(loc="best", fontsize=8)

    def init():
        line.set_data([], [])
        status.set_text("")
        return line, status

    def update(frame: AnimationFrame):
        y_line = [frame.theta0 + (frame.theta1 * x_value) for x_value in x_line]
        line.set_data(x_line, y_line)
        status.set_text(
            f"iter={frame.iteration}\ntheta0={frame.theta0:.2f}\ntheta1={frame.theta1:.6f}\nMSE={frame.mse:.1f}"
        )
        return line, status

    frame_interval_ms = max(1, int(1000 / fps))
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=frame_interval_ms,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def render_and_save(
    mileages: list[float],
    prices: list[float],
    predictions: list[float],
    is_test_flags: list[bool],
    model: Model,
    analysis: PlotAnalysis,
    output_path: Path,
    image_format: str,
    dpi: int,
    show: bool,
    theme_name: str,
    output_color: str | None,
    x_axis: str,
) -> None:
    plt = _import_pyplot(show=show)
    theme = resolve_theme(theme_name)
    apply_theme_style(plt, theme)

    fig, _, _ = _draw_dashboard(
        plt,
        mileages,
        prices,
        predictions,
        is_test_flags,
        model,
        analysis,
        theme,
        output_color,
        x_axis,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=dpi, format=image_format)
    if show:
        plt.show()
    plt.close(fig)
