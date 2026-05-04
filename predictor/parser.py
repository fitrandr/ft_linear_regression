from __future__ import annotations

import math
import re
from pathlib import Path

THOUSANDS_GROUP_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
PLAIN_NUMBER_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")


def normalize_number(value: str) -> str:
    """Normalize numeric string in strict US format."""
    normalized = value.strip()
    if normalized == "":
        raise ValueError("Mileage cannot be empty.")

    compact = normalized.replace(" ", "").replace("_", "")
    if THOUSANDS_GROUP_RE.fullmatch(compact):
        return compact.replace(",", "")
    if PLAIN_NUMBER_RE.fullmatch(compact):
        return compact
    raise ValueError(
        f"Invalid mileage format: {value!r}. Use US format: 12000, 12,000 or 12000.5."
    )


def parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc

    if not math.isfinite(parsed):
        raise ValueError("Value must be finite.")
    return parsed


def parse_mileage(raw_value: str) -> float:
    normalized = normalize_number(raw_value)
    mileage = parse_float(normalized)
    if mileage < 0:
        raise ValueError("Mileage cannot be negative.")
    return mileage


def parse_mileage_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Mileage input file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    values = [line.strip() for line in lines if line.strip()]
    if not values:
        raise ValueError("Mileage input file is empty.")
    return values
