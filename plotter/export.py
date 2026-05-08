from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportImagePaths:
    regression: Path
    residuals: Path
    predicted_vs_actual: Path
    error_distribution: Path


@dataclass(frozen=True)
class PlotExportBundle:
    dashboard: Path
    report_images: ReportImagePaths | None
    animation: Path | None


def _with_format(path: Path, image_format: str) -> Path:
    suffix = f".{image_format.lower()}"
    if path.suffix.lower() == suffix:
        return path
    if path.suffix:
        return path.with_suffix(suffix)
    return path.with_name(path.name + suffix)


def build_report_image_paths(report_dir: Path, image_format: str) -> ReportImagePaths:
    return ReportImagePaths(
        regression=_with_format(report_dir / "regression", image_format),
        residuals=_with_format(report_dir / "residuals", image_format),
        predicted_vs_actual=_with_format(report_dir / "predicted_vs_actual", image_format),
        error_distribution=_with_format(report_dir / "error_distribution", image_format),
    )


def default_animation_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_training.gif")
