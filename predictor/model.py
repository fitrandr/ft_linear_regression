from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_MODEL_VERSION = 1
SUPPORTED_MODEL_VERSIONS = {1}
MAX_ABS_SLOPE = 1_000_000.0
MAX_ABS_INTERCEPT = 1_000_000_000.0
MAX_ABS_PREDICTION = 1_000_000_000_000.0


class ModelPolicy(Enum):
    STRICT = "strict"
    NON_STRICT = "non-strict"


class InputPolicy(Enum):
    SKIP_INVALID = "skip-invalid"
    FAIL_FAST = "fail-fast"


@dataclass(frozen=True)
class Model:
    theta0: float
    theta1: float

    def predict_single(self, mileage: float) -> float:
        return self.theta0 + self.theta1 * mileage


@dataclass(frozen=True)
class PredictArgs:
    model_path: Path
    json_output: bool
    validate_only: bool
    verbose: bool
    quiet: bool
    model_policy: ModelPolicy
    input_policy: InputPolicy
    mileage: str | None
    mileages: tuple[str, ...] | None
    mileages_file: Path | None

    @property
    def fail_fast(self) -> bool:
        return self.input_policy == InputPolicy.FAIL_FAST
