# API Documentation

This reference describes the internal Python API exposed by project modules.

## Overview

The project is split into five module groups:

- `trainer`: training pipeline
- `predictor`: model loading and inference
- `evaluator`: metrics and benchmarking
- `plotter`: visualization and report export
- `interpreter`: plain-language interpretation

## Trainer API

### `trainer.data`

- `load_dataset(path: Path) -> tuple[list[float], list[float]]`
  - Load and validate `km,price` CSV.
- `validate_pairs(mileages: list[float], prices: list[float]) -> None`
  - Ensure aligned non-empty arrays.
- `split_dataset(mileages, prices, test_ratio=0.2, seed=42) -> DatasetSplit`
  - Deterministic split used by trainer/evaluator.

### `trainer.engine`

- `mean(values) -> float`
- `std(values, values_mean) -> float`
- `mse(mileages, prices, model) -> float`
- `train_model(mileages, prices, learning_rate, iterations, log_every, early_stopping_patience, early_stopping_min_delta, logger) -> TrainingResult`

### `trainer.pipeline`

- `run_pipeline(args: TrainArgs, logger: logging.Logger) -> PipelineOutput`
- `build_payload(training_result, metrics, split, test_ratio, seed) -> dict[str, object]`
- `save_model(model_path: Path, payload: dict[str, object]) -> None`

### Core dataclasses (`trainer.model`)

- `LinearModel`
- `TrainingResult`
- `EvaluationMetrics`
- `TrainArgs`
- `PipelineOutput`

## Predictor API

### `predictor.engine`

- `load_model(model_path: Path, model_policy: ModelPolicy, logger) -> tuple[Model, bool]`
  - Returns `(model, used_fallback)`.
- `extract_model(payload: dict[str, Any]) -> Model`
- `predict(model: Model, mileages: list[float]) -> list[float]`
- `assert_model(model: Model) -> None`

### `predictor.parser`

- `normalize_number(value: str) -> str`
- `parse_float(value: str) -> float`
- `parse_mileage(raw_value: str) -> float`
- `parse_mileage_file(path: Path) -> list[str]`

### Core types (`predictor.model`)

- `Model`
- `ModelPolicy` (`STRICT`, `NON_STRICT`)
- `InputPolicy` (`FAIL_FAST`, `SKIP_INVALID`)
- `PredictArgs`

## Evaluator API

### `evaluator.data`

- `load_dataset(path: Path) -> tuple[list[float], list[float]]`
- `validate_dataset(mileages, prices) -> None`

### `evaluator.stats`

- `compute_metrics(prices, predictions) -> RegressionMetrics`
- `compare_with_baseline(prices, model_predictions) -> MetricsComparison`
- `correlation(values_x, values_y) -> float | None`

### `evaluator.pipeline`

- `evaluate(mileages, prices, model, test_ratio, seed) -> EvaluationResult`
- `log_comparison_warnings(logger, scope_name, comparison) -> None`

### `evaluator.report`

- `serialize_result(result: EvaluationResult) -> dict[str, object]`
- `render_text_output(result: EvaluationResult) -> str`
- `save_report(path: Path, result: EvaluationResult) -> None`

### Core dataclasses (`evaluator.model`)

- `RegressionMetrics`
- `MetricsComparison`
- `SplitInfo`
- `EvaluationResult`
- `EvaluateArgs`

## Plotter API

### `plotter.data`

- `load_plot_data(dataset_path, model_path, logger, test_ratio, seed) -> LoadedPlotData`

### `plotter.diagnostics`

- `build_test_flags(sample_count, test_ratio, seed) -> list[bool]`
- `quality_label(r2, usefulness) -> str`
- `predict_minus_actual(predictions, prices) -> list[float]`
- `build_gradient_descent_frames(mileages, prices, iterations, learning_rate=0.1) -> list[AnimationFrame]`

### `plotter.theme`

- `resolve_theme(theme_name) -> PlotTheme`
- `apply_theme_style(plt, theme) -> None`

### `plotter.cli`

- `parse_args() -> PlotArgs`
- `validate_output_color(raw_color: str | None) -> str | None`

### `plotter.export`

- `build_report_image_paths(report_dir, image_format) -> ReportImagePaths`
- `default_animation_path(output_path) -> Path`

### `plotter.report`

- `build_analysis(mileages, prices, predictions, model) -> PlotAnalysis`
- `metrics_annotation(analysis: PlotAnalysis, model: Model) -> str`
- `save_report_bundle(report_dir: Path, plot_path: Path, analysis: PlotAnalysis) -> Path`

### `plotter.render`

- `compute_axis_transform(mileages: list[float], x_axis: str) -> AxisTransform`
- `render_and_save(mileages, prices, predictions, is_test_flags, model, analysis, output_path, image_format, dpi, show, theme_name, output_color, x_axis) -> None`
- `render_report_images(mileages, prices, predictions, is_test_flags, analysis, theme_name, output_color, x_axis, image_paths, dpi) -> None`
- `render_training_animation(mileages, prices, frames, output_path, dpi, fps, theme_name, output_color, is_test_flags) -> None`

### Core dataclasses (`plotter.model`)

- `PlotArgs`
- `LoadedPlotData`
- `PlotAnalysis`
- `AxisTransform`

Additional dataclasses:

- `plotter.theme.PlotTheme`
- `plotter.export.ReportImagePaths`
- `plotter.diagnostics.AnimationFrame`

## Interpreter API

### `interpreter.engine`

- `load_interpreted_report(path: Path) -> InterpretedReport`
- `build_interpretation_text(report: InterpretedReport) -> str`
- `save_interpretation(path: Path, content: str) -> None`

### Core dataclasses (`interpreter.model`)

- `InterpretArgs`
- `ScopeMetrics`
- `InterpretedReport`

## Integration examples

### Programmatic prediction

```python
from pathlib import Path
import logging
from predictor.engine import load_model, predict
from predictor.model import ModelPolicy

logger = logging.getLogger("integration")
model, _ = load_model(Path("report_artifacts/model.json"), ModelPolicy.STRICT, logger)
values = predict(model, [50_000.0, 100_000.0, 150_000.0])
print(values)
```

### Programmatic evaluation

```python
from pathlib import Path
import logging
from predictor.engine import load_model
from predictor.model import ModelPolicy
from evaluator.data import load_dataset
from evaluator.pipeline import evaluate

logger = logging.getLogger("integration")
mileages, prices = load_dataset(Path("data.csv"))
model, _ = load_model(Path("report_artifacts/model.json"), ModelPolicy.STRICT, logger)
result = evaluate(mileages, prices, model, test_ratio=0.2, seed=42)
print(result.test.model.rmse)
```

## Stability contracts

- Loader functions raise `ValueError` on invalid schema/format.
- CLI entrypoints convert fatal errors into `SystemExit(1)`.
- Prediction path enforces finite and bounded outputs.
