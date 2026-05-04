from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from predictor.engine import extract_model, load_model, predict
from predictor.model import Model, ModelPolicy
from predictor.parser import parse_mileage, parse_mileage_file


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test.predictor")
        self.logger.addHandler(logging.NullHandler())

    def test_parse_mileage_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_mileage("-10")

    def test_parse_mileage_strict_us_rejects_eu_decimal(self) -> None:
        with self.assertRaises(ValueError):
            parse_mileage("12,5")

    def test_parse_mileage_accepts_spaces_and_commas(self) -> None:
        self.assertEqual(parse_mileage("12 000"), 12000.0)
        self.assertEqual(parse_mileage("12,000"), 12000.0)

    def test_parse_mileage_file_empty_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mileages.txt"
            path.write_text("\n \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_mileage_file(path)

    def test_extract_model_from_raw_schema(self) -> None:
        payload = {"version": 1, "model": {"theta0": 1000, "theta1": -0.1}}
        model = extract_model(payload)
        self.assertAlmostEqual(model.theta0, 1000.0)
        self.assertAlmostEqual(model.theta1, -0.1)

    def test_extract_model_from_normalized_schema(self) -> None:
        payload = {
            "version": 1,
            "model": {
                "normalized_theta0": 5000.0,
                "normalized_theta1": -2000.0,
                "km_mean": 100000.0,
                "km_std": 50000.0,
            },
        }
        model = extract_model(payload)
        self.assertAlmostEqual(model.theta1, -0.04)

    def test_load_model_strict_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model(Path("/tmp/does_not_exist_model.json"), ModelPolicy.STRICT, self.logger)

    def test_load_model_strict_corrupted_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path, ModelPolicy.STRICT, self.logger)

    def test_predict_out_of_bounds_raises(self) -> None:
        model = Model(theta0=0.0, theta1=10_000_000_000.0)
        with self.assertRaises(ValueError):
            predict(model, [200.0])

    def test_predict_huge_batch(self) -> None:
        model = Model(theta0=1.0, theta1=0.5)
        inputs = [float(i) for i in range(50_000)]
        outputs = predict(model, inputs)
        self.assertEqual(len(outputs), len(inputs))
        self.assertAlmostEqual(outputs[123], 62.5)


if __name__ == "__main__":
    unittest.main()
