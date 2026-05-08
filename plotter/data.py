from __future__ import annotations

import logging

from .diagnostics import build_test_flags
from predictor.engine import load_model, predict
from predictor.model import ModelPolicy
from trainer.data import load_dataset

from .model import LoadedPlotData


def load_plot_data(
    dataset_path,
    model_path,
    logger: logging.Logger,
    test_ratio: float,
    seed: int,
) -> LoadedPlotData:
    mileages, prices = load_dataset(dataset_path)
    model, _ = load_model(model_path, model_policy=ModelPolicy.STRICT, logger=logger)
    predictions = predict(model, mileages)
    is_test_flags = build_test_flags(len(mileages), test_ratio=test_ratio, seed=seed)
    return LoadedPlotData(
        mileages=mileages,
        prices=prices,
        model=model,
        predictions=predictions,
        is_test_flags=is_test_flags,
    )
