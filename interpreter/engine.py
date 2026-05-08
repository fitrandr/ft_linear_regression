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
        return "non defini"
    return f"{value:.{precision}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "non calcule"
    return f"{value * 100.0:.1f}%"


def _scope_interpretation_simple(scope: ScopeMetrics) -> list[str]:
    verdict = "meilleur" if scope.delta_mse > 0 else "moins bon"
    lines = [f"- {scope.name.upper()}:"]
    lines.append(f"  - Le modele est {verdict} qu'une methode tres simple (prix moyen).")
    lines.append(f"  - Erreur moyenne: environ {scope.model_mae:.0f} euros.")
    lines.append(f"  - Erreur typique: environ {scope.model_rmse:.0f} euros.")

    if scope.model_r2_defined and scope.model_r2 is not None:
        lines.append(
            "  - Indice de fiabilite globale (R2): "
            f"{scope.model_r2:.2f} (plus proche de 1 = mieux)."
        )
    else:
        lines.append("  - Indice de fiabilite globale (R2): non defini.")

    lines.append(f"  - Gain par rapport a la baseline: {_pct(scope.usefulness_score)}.")
    lines.append(f"  - Points tres atypiques detectes: {scope.model_outlier_count}.")
    return lines


def _generalization_note(overfit_gap: float | None) -> str:
    if overfit_gap is None:
        return "- Ecart train/test: non defini."
    if overfit_gap > 0.20:
        return (
            f"- Ecart train/test: {overfit_gap:.3f}. "
            "Le modele parait bien meilleur en entrainement qu'en test (risque de surapprentissage)."
        )
    if overfit_gap > 0.08:
        return (
            f"- Ecart train/test: {overfit_gap:.3f}. "
            "Le comportement est correct mais a surveiller."
        )
    return f"- Ecart train/test: {overfit_gap:.3f}. Le comportement train/test est stable."


def _correlation_note(correlation: float | None) -> str:
    if correlation is None:
        return "- Lien kilometrage/prix: non defini."
    strength = abs(correlation)
    if strength >= 0.80:
        label = "tres fort"
    elif strength >= 0.60:
        label = "fort"
    elif strength >= 0.40:
        label = "moyen"
    elif strength >= 0.20:
        label = "faible"
    else:
        label = "tres faible"

    if correlation < 0:
        direction_sentence = "quand le kilometrage monte, le prix baisse."
    else:
        direction_sentence = "quand le kilometrage monte, le prix monte."
    return f"- Lien kilometrage/prix: {label}; {direction_sentence}"


def build_interpretation_text(report: InterpretedReport) -> str:
    lines: list[str] = []
    lines.append("Rapport d'interpretation (version simple)")
    lines.append("=" * 40)
    lines.append(
        "Objectif: expliquer avec des mots simples si le modele est utile "
        "pour estimer le prix."
    )
    lines.append(
        f"Echantillons: {report.samples} (train={report.train_samples}, "
        f"test={report.test_samples}, ratio test={report.test_ratio}, seed={report.seed})"
    )
    lines.append("")

    if report.test.delta_mse <= 0:
        resume = "Le modele n'est pas encore fiable: il fait pire que la baseline sur le test."
    elif report.test.usefulness_score is not None and report.test.usefulness_score < 0.30:
        resume = "Le modele est utilisable, mais le gain reste faible."
    else:
        resume = "Le modele est globalement utile pour une estimation rapide."
    lines.append(f"Resume rapide: {resume}")

    lines.append("")
    lines.append("Lecture simple des resultats")
    lines.extend(_scope_interpretation_simple(report.full))
    lines.extend(_scope_interpretation_simple(report.train))
    lines.extend(_scope_interpretation_simple(report.test))

    lines.append("")
    lines.append("Ce qu'il faut retenir")
    lines.append(_generalization_note(report.overfit_gap))
    lines.append(_correlation_note(report.correlation))

    lines.append("")
    lines.append("Mini dictionnaire")
    lines.append("- MAE: erreur moyenne (en euros). Plus petit = mieux.")
    lines.append("- RMSE: erreur typique. Plus petit = mieux.")
    lines.append("- R2: niveau de fiabilite globale. Plus proche de 1 = mieux.")

    lines.append("")
    lines.append("Conseil pratique")
    if report.test.delta_mse <= 0:
        lines.append(
            "- Ne pas utiliser ce modele en production pour l'instant. "
            "Il faut l'ameliorer (donnees, reglages, variables)."
        )
    elif report.test.usefulness_score is not None and report.test.usefulness_score < 0.30:
        lines.append(
            "- Utilisable pour une premiere estimation, mais pas pour une decision finale."
        )
    else:
        lines.append(
            "- Bon outil pour une estimation rapide. "
            "Toujours verifier avec d'autres informations du vehicule."
        )
    return "\n".join(lines) + "\n"


def save_interpretation(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
