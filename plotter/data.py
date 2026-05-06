from __future__ import annotations

import logging

from predictor.engine import load_model, predict
from predictor.model import ModelPolicy
from trainer.data import load_dataset

from .model import LoadedPlotData


def load_plot_data(dataset_path, model_path, logger: logging.Logger) -> LoadedPlotData:
    mileages, prices = load_dataset(dataset_path)
    model, _ = load_model(model_path, model_policy=ModelPolicy.STRICT, logger=logger)
    predictions = predict(model, mileages)
    return LoadedPlotData(
        mileages=mileages,
        prices=prices,
        model=model,
        predictions=predictions,
    )
