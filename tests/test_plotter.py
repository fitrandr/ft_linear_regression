from __future__ import annotations

import unittest
from pathlib import Path

from plotter.cli import resolve_output_base
from plotter.render import compute_axis_transform
from plotter.report import build_analysis, resolve_output_path


class PlotterTests(unittest.TestCase):
    def test_resolve_output_base_uses_report_dir_for_relative_name(self) -> None:
        resolved = resolve_output_base("regression_plot_make", Path("report_artifacts"))
        self.assertEqual(resolved, Path("report_artifacts/regression_plot_make"))

    def test_resolve_output_path_adds_extension(self) -> None:
        self.assertEqual(
            resolve_output_path(Path("regression_plot"), "png"),
            Path("regression_plot.png"),
        )
        self.assertEqual(
            resolve_output_path(Path("plot.jpeg"), "svg"),
            Path("plot.svg"),
        )

    def test_compute_axis_transform_normalized(self) -> None:
        axis = compute_axis_transform([1000.0, 2000.0, 3000.0], "normalized")
        self.assertTrue(axis.normalized)
        self.assertEqual(len(axis.x_values), 3)

    def test_compute_axis_transform_raises_when_no_variance(self) -> None:
        with self.assertRaises(ValueError):
            compute_axis_transform([1000.0, 1000.0], "normalized")

    def test_build_analysis_detects_outlier_flag(self) -> None:
        mileages = [float(1000 * (idx + 1)) for idx in range(25)]
        prices = [10000.0 - (200.0 * idx) for idx in range(25)]
        predictions = list(prices)
        predictions[12] = prices[12] - 3000.0

        analysis = build_analysis(mileages, prices, predictions)
        self.assertEqual(len(analysis.residuals), len(prices))
        self.assertEqual(len(analysis.outlier_flags), len(prices))
        self.assertGreaterEqual(analysis.outlier_count, 1)


if __name__ == "__main__":
    unittest.main()
