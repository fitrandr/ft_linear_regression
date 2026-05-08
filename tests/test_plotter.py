from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from predictor.model import Model
from plotter.cli import resolve_output_base
from plotter.diagnostics import build_gradient_descent_frames, build_test_flags
from plotter.export import build_report_image_paths
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
        model = Model(theta0=9000.0, theta1=-0.2)

        analysis = build_analysis(mileages, prices, predictions, model)
        self.assertEqual(len(analysis.residuals), len(prices))
        self.assertEqual(len(analysis.outlier_flags), len(prices))
        self.assertGreaterEqual(analysis.outlier_count, 1)
        self.assertIn(analysis.quality_label, {"excellent", "good", "fair", "weak", "undefined", "worse-than-baseline"})

        annotation = metrics_annotation(analysis, model)
        self.assertIn("Samples:", annotation)
        self.assertIn("Residual mean", annotation)
        self.assertIn("Mileage range:", annotation)
        self.assertIn("Slope(theta1)", annotation)

    def test_save_report_bundle_writes_descriptive_summary(self) -> None:
        mileages = [1000.0, 2000.0, 3000.0, 4000.0]
        prices = [10000.0, 9000.0, 8000.0, 7000.0]
        predictions = [9900.0, 9100.0, 7900.0, 7100.0]
        model = Model(theta0=11000.0, theta1=-1.0)
        analysis = build_analysis(mileages, prices, predictions, model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_dir = Path(tmp_dir)
            metrics_path = save_report_bundle(report_dir, Path("report_artifacts/regression_plot_make.png"), analysis)
            self.assertTrue(metrics_path.exists())

            summary_text = (report_dir / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Plot Summary Report", summary_text)
            self.assertIn("Residual diagnostics", summary_text)
            self.assertIn("Usefulness score", summary_text)
            self.assertIn("Model quality", summary_text)

    def test_build_test_flags_has_expected_size(self) -> None:
        flags = build_test_flags(10, test_ratio=0.2, seed=42)
        self.assertEqual(len(flags), 10)
        self.assertGreaterEqual(sum(flags), 1)
        self.assertLess(sum(flags), 10)

    def test_build_gradient_descent_frames_returns_iterations(self) -> None:
        mileages = [1000.0, 2000.0, 3000.0, 4000.0]
        prices = [10000.0, 9200.0, 8300.0, 7400.0]
        frames = build_gradient_descent_frames(mileages, prices, iterations=20, learning_rate=0.1)
        self.assertEqual(len(frames), 20)
        self.assertEqual(frames[0].iteration, 1)
        self.assertEqual(frames[-1].iteration, 20)

    def test_build_report_image_paths_uses_format(self) -> None:
        paths = build_report_image_paths(Path("report_artifacts"), "svg")
        self.assertEqual(paths.regression.suffix, ".svg")
        self.assertEqual(paths.residuals.suffix, ".svg")
        self.assertEqual(paths.predicted_vs_actual.suffix, ".svg")
        self.assertEqual(paths.error_distribution.suffix, ".svg")


if __name__ == "__main__":
    unittest.main()
