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


def _pct(value: float | None) -> str:
    if value is None:
        return "not computed"
    return f"{value * 100.0:.1f}%"


def _scope_interpretation_simple(scope: ScopeMetrics) -> list[str]:
    verdict = "better" if scope.delta_mse > 0 else "worse"
    lines = [f"- {scope.name.upper()}:"]
    lines.append(f"  - The model is {verdict} than a very simple baseline (mean price).")
    lines.append(f"  - Average error: around {scope.model_mae:.0f} euros.")
    lines.append(f"  - Typical error: around {scope.model_rmse:.0f} euros.")

    if scope.model_r2_defined and scope.model_r2 is not None:
        lines.append(
            "  - Global fit indicator (R2): "
            f"{scope.model_r2:.2f} (closer to 1 is better)."
        )
    else:
        lines.append("  - Global fit indicator (R2): undefined.")

    lines.append(f"  - Improvement vs baseline: {_pct(scope.usefulness_score)}.")
    lines.append(f"  - Strongly unusual points detected: {scope.model_outlier_count}.")
    return lines


def _generalization_note(overfit_gap: float | None) -> str:
    if overfit_gap is None:
        return "- Train/test gap: undefined."
    if overfit_gap > 0.20:
        return (
            f"- Train/test gap: {overfit_gap:.3f}. "
            "The model looks much better on training than on test data (possible overfitting risk)."
        )
    if overfit_gap > 0.08:
        return (
            f"- Train/test gap: {overfit_gap:.3f}. "
            "Generalization is acceptable but should be monitored."
        )
    return f"- Train/test gap: {overfit_gap:.3f}. Train/test behavior is stable."


def _correlation_note(correlation: float | None) -> str:
    if correlation is None:
        return "- Mileage/price relationship: undefined."
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

    if correlation < 0:
        direction_sentence = "when mileage goes up, price usually goes down."
    else:
        direction_sentence = "when mileage goes up, price usually goes up."
    return f"- Mileage/price relationship: {label}; {direction_sentence}"


def build_interpretation_text(report: InterpretedReport) -> str:
    lines: list[str] = []
    lines.append("Interpretation Report (Simple English)")
    lines.append("=" * 40)
    lines.append(
        "Goal: explain in simple words whether the model is useful "
        "for estimating a car price."
    )
    lines.append(
        f"Samples: {report.samples} (train={report.train_samples}, "
        f"test={report.test_samples}, ratio test={report.test_ratio}, seed={report.seed})"
    )
    lines.append("")

    if report.test.delta_mse <= 0:
        summary = "The model is not reliable yet: it performs worse than baseline on test data."
    elif report.test.usefulness_score is not None and report.test.usefulness_score < 0.30:
        summary = "The model is usable, but the improvement is still limited."
    else:
        summary = "The model is overall useful for quick estimates."
    lines.append(f"Quick summary: {summary}")

    lines.append("")
    lines.append("Simple reading of results")
    lines.extend(_scope_interpretation_simple(report.full))
    lines.extend(_scope_interpretation_simple(report.train))
    lines.extend(_scope_interpretation_simple(report.test))

    lines.append("")
    lines.append("What to remember")
    lines.append(_generalization_note(report.overfit_gap))
    lines.append(_correlation_note(report.correlation))

    lines.append("")
    lines.append("Quick glossary")
    lines.append("- MAE: average error (in euros). Smaller is better.")
    lines.append("- RMSE: typical error size. Smaller is better.")
    lines.append("- R2: overall fit quality. Closer to 1 is better.")

    lines.append("")
    lines.append("Practical advice")
    if report.test.delta_mse <= 0:
        lines.append(
            "- Do not use this model in production yet. "
            "Improve data quality, tuning, or input features first."
        )
    elif report.test.usefulness_score is not None and report.test.usefulness_score < 0.30:
        lines.append(
            "- Useful for a first estimate, but not for a final decision."
        )
    else:
        lines.append(
            "- Good tool for a quick estimate. "
            "Always cross-check with other vehicle information."
        )
    return "\n".join(lines) + "\n"


def save_interpretation(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
