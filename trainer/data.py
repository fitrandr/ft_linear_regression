from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from .model import DatasetSplit


def load_dataset(path: Path) -> tuple[list[float], list[float]]:
    mileages: list[float] = []
    prices: list[float] = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        raw_headers = reader.fieldnames or []
        header_map: dict[str, str] = {}
        for header in raw_headers:
            if header is not None:
                header_map[header.strip()] = header

        if "km" not in header_map or "price" not in header_map:
            raise ValueError("CSV must contain 'km' and 'price' columns.")

        km_key = header_map["km"]
        price_key = header_map["price"]

        for line_no, row in enumerate(reader, start=2):
            raw_km = (row.get(km_key) or "").strip()
            raw_price = (row.get(price_key) or "").strip()

            if raw_km == "" or raw_price == "":
                raise ValueError(f"Missing value at line {line_no}: {row}")

            try:
                km = float(raw_km)
                price = float(raw_price)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value at line {line_no}: {row}") from exc

            if not math.isfinite(km) or not math.isfinite(price):
                raise ValueError(f"Non-finite numeric value at line {line_no}: {row}")

            mileages.append(km)
            prices.append(price)

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def validate_pairs(mileages: list[float], prices: list[float]) -> None:
    if not mileages or not prices:
        raise ValueError("mileages and prices must not be empty.")
    if len(mileages) != len(prices):
        raise ValueError("mileages and prices must have the same number of samples.")


def split_dataset(
    mileages: list[float],
    prices: list[float],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    validate_pairs(mileages, prices)
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in the interval (0, 1).")

    m = len(mileages)
    if m < 2:
        raise ValueError("At least 2 samples are required for train/test split.")

    indices = list(range(m))
    random.Random(seed).shuffle(indices)

    test_size = max(1, int(m * test_ratio))
    if test_size >= m:
        test_size = m - 1

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_mileages = [mileages[i] for i in train_indices]
    train_prices = [prices[i] for i in train_indices]
    test_mileages = [mileages[i] for i in test_indices]
    test_prices = [prices[i] for i in test_indices]
    return train_mileages, train_prices, test_mileages, test_prices
