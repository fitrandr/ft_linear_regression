from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotTheme:
    name: str
    background: str
    axes_background: str
    text: str
    grid: str
    train: str
    test: str
    outlier: str
    regression: str
    baseline: str
    residual_line: str


LIGHT_THEME = PlotTheme(
    name="light",
    background="#ffffff",
    axes_background="#ffffff",
    text="#111827",
    grid="#cbd5e1",
    train="#1d4ed8",
    test="#f59e0b",
    outlier="#dc2626",
    regression="#7c3aed",
    baseline="#64748b",
    residual_line="#94a3b8",
)

DARK_THEME = PlotTheme(
    name="dark",
    background="#0f172a",
    axes_background="#111827",
    text="#f8fafc",
    grid="#334155",
    train="#60a5fa",
    test="#fbbf24",
    outlier="#f87171",
    regression="#c084fc",
    baseline="#94a3b8",
    residual_line="#64748b",
)


def resolve_theme(theme_name: str) -> PlotTheme:
    if theme_name == "dark":
        return DARK_THEME
    return LIGHT_THEME


def apply_theme_style(plt, theme: PlotTheme) -> None:
    if theme.name == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
