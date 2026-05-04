from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from trainer.data import load_dataset, split_dataset
from trainer.engine import train_model
from trainer.model import TrainArgs
from trainer.pipeline import run_pipeline


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test.trainer")
        self.logger.addHandler(logging.NullHandler())

    def test_load_dataset_missing_columns_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("x,y\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_split_dataset_counts(self) -> None:
        mileages = [float(i) for i in range(10)]
        prices = [1000.0 - 10.0 * i for i in range(10)]
        train_x, train_y, test_x, test_y = split_dataset(mileages, prices, test_ratio=0.2, seed=42)
        self.assertEqual(len(train_x), len(train_y))
        self.assertEqual(len(test_x), len(test_y))
        self.assertEqual(len(train_x) + len(test_x), 10)

    def test_train_model_runs(self) -> None:
        mileages = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        prices = [9000.0, 8000.0, 7000.0, 6000.0, 5000.0]
        result = train_model(
            mileages,
            prices,
            learning_rate=0.1,
            iterations=1000,
            log_every=200,
            early_stopping_patience=80,
            early_stopping_min_delta=1e-8,
            logger=self.logger,
        )
        self.assertLessEqual(result.iterations_ran, result.iterations)
        self.assertTrue(result.model.theta1 < 0)

    def test_run_pipeline_writes_versioned_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "data.csv"
            model = Path(tmp) / "model.json"
            dataset.write_text(
                "km,price\n10000,9000\n20000,8000\n30000,7000\n40000,6000\n50000,5000\n",
                encoding="utf-8",
            )
            args = TrainArgs(
                dataset_path=dataset,
                model_path=model,
                learning_rate=0.1,
                iterations=500,
                test_ratio=0.2,
                seed=42,
                log_every=100,
                early_stopping_patience=60,
                early_stopping_min_delta=1e-8,
                verbosity="quiet",
            )
            output = run_pipeline(args, self.logger)
            self.assertTrue(output.model_path.exists())
            content = output.model_path.read_text(encoding="utf-8")
            self.assertIn('"version": 1', content)
            self.assertIn('"model"', content)


if __name__ == "__main__":
    unittest.main()
