# Architecture

This document describes the internal design of `ft_linear_regression`.

## Layered architecture

```mermaid
flowchart LR
    subgraph CLI
        T[train.py]
        P[predict.py]
        E[evaluate.py]
        G[plot.py]
        I[interpret.py]
    end

    subgraph Modules
        TR[trainer]
        PR[predictor]
        EV[evaluator]
        PL[plotter cli/data/render]
        PD[plotter diagnostics/theme/export]
        IN[interpreter]
    end

    subgraph Artifacts
        M[model.json]
        R[evaluation_report.json]
        TXT[interpretation_report.txt]
        IMG[regression_plot_dashboard.png]
        IMGS[regression+residuals+actual_vs_pred+hist]
        GIF[training_animation.gif]
        MET[metrics.json/summary.txt]
    end

    T --> TR --> M
    P --> PR --> M
    E --> EV --> M
    EV --> R
    I --> IN --> R
    IN --> TXT
    G --> PL --> M
    PL --> PD
    PL --> IMG
    PL --> IMGS
    PD --> GIF
    PL --> MET
```

## Data flow

1. `trainer` loads dataset, validates values, splits train/test, trains model, and writes payload.
2. `predictor` loads/validates model payload and predicts in safe bounded mode.
3. `evaluator` computes model and baseline metrics on full/train/test scopes.
4. `plotter` renders a 2x2 dashboard (regression, residuals, histogram, actual-vs-predicted), can export per-diagnostic images, and can generate an optional gradient descent animation GIF.
5. `interpreter` converts evaluation JSON into plain-language conclusions.

## Key design principles

- Explicit validation at boundaries (CSV, JSON, CLI input).
- Typed data flow through dataclasses.
- Small focused modules over monolithic scripts.
- Predictable fallback and strict policies.

## Boundaries and responsibilities

- `trainer`: training-only logic and model payload generation.
- `predictor`: runtime model loading and prediction path.
- `evaluator`: metric math and benchmark comparison.
- `plotter`: visual diagnostics and report extraction.
- `interpreter`: communication layer for non-technical readers.

## Reliability notes

- Normalization + epsilon checks in training.
- Finite checks for all numeric inputs.
- Prediction upper-bound guard against unstable model states.
- Non-zero exit codes for fatal CLI errors.
