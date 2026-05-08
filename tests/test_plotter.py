from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from plotter.cli import resolve_output_base
from plotter.render import compute_axis_transform
from plotter.report import build_analysis, metrics_annotation, resolve_output_path, save_report_bundle


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

        annotation = metrics_annotation(analysis)
        self.assertIn("Samples:", annotation)
        self.assertIn("Residual mean", annotation)
        self.assertIn("Mileage range:", annotation)

    def test_save_report_bundle_writes_descriptive_summary(self) -> None:
        mileages = [1000.0, 2000.0, 3000.0, 4000.0]
        prices = [10000.0, 9000.0, 8000.0, 7000.0]
        predictions = [9900.0, 9100.0, 7900.0, 7100.0]
        analysis = build_analysis(mileages, prices, predictions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            metrics_path = save_report_bundle(report_dir, Path("report_artifacts/regression_plot_make.png"), analysis)
            self.assertTrue(metrics_path.exists())

            summary_text = (report_dir / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Plot Summary Report", summary_text)
            self.assertIn("Residual diagnostics", summary_text)
            self.assertIn("Usefulness score", summary_text)


if __name__ == "__main__":
    unittest.main()
