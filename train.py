#!/usr/bin/env ./env/bin/python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def load_dataset(path: Path) -> tuple[list[float], list[float]]:
    """
    Load a CSV dataset containing car mileage and price information.

    The CSV file must contain two columns:
        - km
        - price

    Each row is read using csv.DictReader, which converts
    the CSV rows into dictionaries using the header names
    as keys.

    Example CSV:
        km,price
        24000,15000
        35000,12000

    Example row produced by DictReader:
        {
            "km": "24000",
            "price": "15000"
        }

    The function:
        1. Opens the CSV file
        2. Checks that required columns exist
        3. Converts values to float
        4. Stores them into two separate lists
        5. Returns both lists

    Args:
        path (Path):
            Path to the CSV dataset file.

    Returns:
        tuple[list[float], list[float]]:
            - First list contains mileage values
            - Second list contains price values

    Raises:
        ValueError:
            If the CSV does not contain 'km'
            or 'price' columns.

        ValueError:
            If the dataset is empty.
    """

    mileages: list[float] = []
    prices: list[float] = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if "km" not in reader.fieldnames or "price" not in reader.fieldnames:
            raise ValueError("CSV must contain 'km' and 'price' columns.")
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


