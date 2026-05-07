from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import InterpretedReport, ScopeMetrics


def _as_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid report: expected object at '{context}'.")
    return data


def _as_float(value: Any, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid report: expected numeric value at '{context}'.") from exc


def _as_bool(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Invalid report: expected boolean value at '{context}'.")


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid report: expected integer value at '{context}'.")
    if isinstance(value, int):
        return value
    raise ValueError(f"Invalid report: expected integer value at '{context}'.")


def _optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, context)


def _read_scope(name: str, payload: dict[str, Any]) -> ScopeMetrics:
    scope = _as_mapping(payload, name)
    model = _as_mapping(scope.get("model"), f"{name}.model")
    baseline = _as_mapping(scope.get("baseline"), f"{name}.baseline")

    return ScopeMetrics(
        name=name,
        model_mae=_as_float(model.get("mae"), f"{name}.model.mae"),
        model_rmse=_as_float(model.get("rmse"), f"{name}.model.rmse"),
        model_mse=_as_float(model.get("mse"), f"{name}.model.mse"),
        model_r2=_optional_float(model.get("r2"), f"{name}.model.r2"),
        model_r2_defined=_as_bool(model.get("r2_defined"), f"{name}.model.r2_defined"),
        model_mean_error=_as_float(model.get("mean_error"), f"{name}.model.mean_error"),
        model_outlier_count=_as_int(model.get("outlier_count"), f"{name}.model.outlier_count"),
        baseline_mse=_as_float(baseline.get("mse"), f"{name}.baseline.mse"),
        baseline_rmse=_as_float(baseline.get("rmse"), f"{name}.baseline.rmse"),
        delta_mse=_as_float(scope.get("delta_mse"), f"{name}.delta_mse"),
        signal_to_noise_ratio=_optional_float(
            scope.get("signal_to_noise_ratio"), f"{name}.signal_to_noise_ratio"
        ),
        usefulness_score=_optional_float(scope.get("usefulness_score"), f"{name}.usefulness_score"),
    )


def load_interpreted_report(path: Path) -> InterpretedReport:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid evaluation report file.") from exc

    root = _as_mapping(raw, "root")
    split = _as_mapping(root.get("split"), "split")

    full_scope = _read_scope("full", _as_mapping(root.get("full"), "full"))
    train_scope = _read_scope("train", _as_mapping(root.get("train"), "train"))
    test_scope = _read_scope("test", _as_mapping(root.get("test"), "test"))

    train_usefulness = train_scope.usefulness_score
    test_usefulness = test_scope.usefulness_score
    overfit_gap: float | None
    if train_usefulness is None or test_usefulness is None:
        overfit_gap = None
    else:
        overfit_gap = train_usefulness - test_usefulness

    return InterpretedReport(
        samples=_as_int(root.get("samples"), "samples"),
        train_samples=_as_int(split.get("train_samples"), "split.train_samples"),
        test_samples=_as_int(split.get("test_samples"), "split.test_samples"),
        test_ratio=_as_float(split.get("test_ratio"), "split.test_ratio"),
        seed=_as_int(split.get("seed"), "split.seed"),
        correlation=_optional_float(root.get("mileage_price_correlation"), "mileage_price_correlation"),
        full=full_scope,
        train=train_scope,
        test=test_scope,
        overfit_gap=overfit_gap,
    )


def _fmt(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "undefined"
    return f"{value:.{precision}f}"


def _quality_label(usefulness: float | None, r2_defined: bool, r2: float | None) -> str:
    if usefulness is None:
        return "indeterminate"
    if usefulness < 0:
        return "poor"
    if not r2_defined or r2 is None:
        return "limited"
    if usefulness >= 0.70 and r2 >= 0.70:
        return "strong"
    if usefulness >= 0.45 and r2 >= 0.45:
        return "good"
    if usefulness >= 0.20 and r2 >= 0.20:
        return "moderate"
    return "weak"


def _scope_interpretation(scope: ScopeMetrics) -> list[str]:
    label = _quality_label(scope.usefulness_score, scope.model_r2_defined, scope.model_r2)
    verdict = "better" if scope.delta_mse > 0 else "worse"
    lines = [
        f"- {scope.name.upper()}: model is {verdict} than baseline (ΔMSE={scope.delta_mse:.3f}).",
        f"  RMSE={scope.model_rmse:.3f}, MAE={scope.model_mae:.3f}, R2={_fmt(scope.model_r2)}.",
        f"  Usefulness={_fmt(scope.usefulness_score)}, SNR={_fmt(scope.signal_to_noise_ratio)}, quality={label}.",
        f"  Mean error={scope.model_mean_error:.3f}, outliers={scope.model_outlier_count}.",
    ]
    return lines


def _generalization_note(overfit_gap: float | None) -> str:
    if overfit_gap is None:
        return "- Generalization gap: undefined (insufficient data)."
    if overfit_gap > 0.20:
        return f"- Generalization gap: {overfit_gap:.4f} (high; possible overfitting)."
    if overfit_gap > 0.08:
        return f"- Generalization gap: {overfit_gap:.4f} (moderate; monitor overfitting)."
    return f"- Generalization gap: {overfit_gap:.4f} (stable train/test behavior)."


def _correlation_note(correlation: float | None) -> str:
    if correlation is None:
        return "- Correlation(km, price): undefined."
    strength = abs(correlation)
    if strength >= 0.80:
        label = "very strong"
    elif strength >= 0.60:
        label = "strong"
    elif strength >= 0.40:
        label = "moderate"
    elif strength >= 0.20:
        label = "weak"
    else:
        label = "very weak"
    direction = "negative" if correlation < 0 else "positive"
    return f"- Correlation(km, price): {correlation:.4f} ({label} {direction})."


def build_interpretation_text(report: InterpretedReport) -> str:
    lines: list[str] = []
    lines.append("Model Interpretation Report")
    lines.append("=" * 28)
    lines.append(
        f"Samples: {report.samples} (train={report.train_samples}, test={report.test_samples}, "
        f"ratio={report.test_ratio}, seed={report.seed})"
    )
    lines.append("")
    lines.append("Scope Analysis")
    lines.extend(_scope_interpretation(report.full))
    lines.extend(_scope_interpretation(report.train))
    lines.extend(_scope_interpretation(report.test))
    lines.append("")
    lines.append("General Diagnostics")
    lines.append(_generalization_note(report.overfit_gap))
    lines.append(_correlation_note(report.correlation))

    if report.test.delta_mse <= 0:
        action = "Model does not beat baseline on test data; revisit features or hyperparameters."
    elif report.test.usefulness_score is not None and report.test.usefulness_score < 0.30:
        action = "Model beats baseline but with low margin; consider richer features or robust tuning."
    else:
        action = "Model shows useful predictive signal on test data."

    lines.append("")
    lines.append("Conclusion")
    lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def save_interpretation(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
