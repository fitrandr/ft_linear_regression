from __future__ import annotations

import csv
import math
from pathlib import Path

from .model import EPSILON


def _parse_finite_float(raw_value: str, field: str, line_no: int) -> float:
    value = raw_value.strip()
    if value == "":
        raise ValueError(f"Missing {field} value at line {line_no}.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric {field} value at line {line_no}: {raw_value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value at line {line_no}: {raw_value!r}.")
    return parsed


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
            raw_km = row.get(km_key)
            raw_price = row.get(price_key)
            if raw_km is None or raw_price is None:
                raise ValueError(f"Missing column value at line {line_no}: {row}")

            km = _parse_finite_float(raw_km, field="km", line_no=line_no)
            price = _parse_finite_float(raw_price, field="price", line_no=line_no)
            mileages.append(km)
            prices.append(price)

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def _validate_finite_series(name: str, values: list[float]) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name} at index {index}.")


def validate_dataset(mileages: list[float], prices: list[float]) -> None:
    if not mileages or not prices:
        raise ValueError("Empty dataset.")
    if len(mileages) != len(prices):
        raise ValueError("Mismatched dataset sizes.")

    _validate_finite_series("mileage", mileages)
    _validate_finite_series("price", prices)

    mileage_span = max(mileages) - min(mileages)
    if mileage_span < EPSILON:
        raise ValueError("Not enough variance in mileage.")
