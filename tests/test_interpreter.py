from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interpreter.engine import (
    build_interpretation_text,
    load_interpreted_report,
    save_interpretation,
)


def _valid_report_payload() -> dict[str, object]:
    return {
        "samples": 10,
        "full": {
            "model": {
                "mae": 100.0,
                "mse": 20000.0,
                "rmse": 141.421,
                "r2": 0.8,
                "r2_defined": True,
                "mean_error": -10.0,
                "error_std": 120.0,
                "max_abs_error": 300.0,
                "outlier_count": 1,
            },
            "baseline": {
                "mae": 200.0,
                "mse": 50000.0,
                "rmse": 223.606,
                "r2": 0.0,
                "r2_defined": True,
                "mean_error": 0.0,
                "error_std": 223.606,
                "max_abs_error": 500.0,
                "outlier_count": 2,
            },
            "delta_mse": 30000.0,
            "signal_to_noise_ratio": 2.5,
            "usefulness_score": 0.6,
        },
        "train": {
            "model": {
                "mae": 90.0,
                "mse": 18000.0,
                "rmse": 134.164,
                "r2": 0.82,
                "r2_defined": True,
                "mean_error": -8.0,
                "error_std": 110.0,
                "max_abs_error": 250.0,
                "outlier_count": 1,
            },
            "baseline": {
                "mae": 210.0,
                "mse": 52000.0,
                "rmse": 228.035,
                "r2": 0.0,
                "r2_defined": True,
                "mean_error": 0.0,
                "error_std": 228.035,
                "max_abs_error": 540.0,
                "outlier_count": 2,
            },
            "delta_mse": 34000.0,
            "signal_to_noise_ratio": 2.88,
            "usefulness_score": 0.6538,
        },
        "test": {
            "model": {
                "mae": 120.0,
                "mse": 25000.0,
                "rmse": 158.114,
                "r2": 0.7,
                "r2_defined": True,
                "mean_error": -20.0,
                "error_std": 130.0,
                "max_abs_error": 320.0,
                "outlier_count": 0,
            },
            "baseline": {
                "mae": 205.0,
                "mse": 48000.0,
                "rmse": 219.089,
                "r2": 0.0,
                "r2_defined": True,
                "mean_error": 0.0,
                "error_std": 219.089,
                "max_abs_error": 510.0,
                "outlier_count": 1,
            },
            "delta_mse": 23000.0,
            "signal_to_noise_ratio": 1.92,
            "usefulness_score": 0.4792,
        },
        "split": {
            "test_ratio": 0.2,
            "seed": 42,
            "train_samples": 8,
            "test_samples": 2,
        },
        "mileage_price_correlation": -0.9,
    }


class InterpreterTests(unittest.TestCase):
    def test_load_and_interpret_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "evaluation_report.json"
            report_path.write_text(json.dumps(_valid_report_payload()), encoding="utf-8")

            interpreted = load_interpreted_report(report_path)
            self.assertEqual(interpreted.samples, 10)
            self.assertAlmostEqual(interpreted.overfit_gap or 0.0, 0.1746, places=3)

            text = build_interpretation_text(interpreted)
            self.assertIn("Interpretation Report", text)
            self.assertIn("FULL", text)
            self.assertIn("Quick summary", text)
            self.assertIn("Quick glossary", text)

    def test_invalid_report_missing_scope_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "bad_report.json"
            payload = _valid_report_payload()
            payload.pop("test")
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_interpreted_report(report_path)

    def test_save_interpretation_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "interpretation.txt"
            save_interpretation(out_path, "hello\n")
            self.assertEqual(out_path.read_text(encoding="utf-8"), "hello\n")


if __name__ == "__main__":
    unittest.main()
