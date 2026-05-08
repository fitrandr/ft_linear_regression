# ft_linear_regression

A production-style, educational linear regression project implemented in pure Python.

This repository was built to learn and demonstrate how a complete ML workflow works without NumPy, pandas, or external ML frameworks.

## Why this project exists

Most linear regression tutorials hide details inside libraries. This project does the opposite:

- We implement data loading and validation ourselves
- We implement gradient descent ourselves
- We keep model serialization explicit and inspectable
- We provide CLI tools for training, prediction, evaluation, plotting, and report interpretation

The goal is to understand every layer of a small ML system, from raw CSV to human-readable report.

## Linear regression explained (simple)

We try to predict a car price from mileage using a straight line:

\[
\hat{y} = \theta_0 + \theta_1 x
\]

- `x`: mileage (km)
- `\hat{y}`: predicted price
- `\theta_0`: intercept
- `\theta_1`: slope

During training:

- We normalize mileage to improve numerical stability
- We optimize `theta0` and `theta1` with gradient descent
- We monitor loss (MSE)
- We support early stopping to avoid unnecessary iterations
- We convert normalized parameters back to raw-scale parameters for prediction

During evaluation:

- We compare model errors against a baseline (predicting the mean price)
- We compute MAE, MSE, RMSE, R2, residual stats, and usefulness indicators
- We evaluate full dataset and train/test split metrics

## Project architecture

### Entrypoints

- `train.py` -> train model and save `model.json`
- `predict.py` -> predict one or multiple prices
- `evaluate.py` -> compute model quality metrics
- `plot.py` -> generate diagnostic charts and plot metrics
- `interpret.py` -> write a plain-language interpretation report

Each entrypoint delegates business logic to a package:

- `trainer/`
- `predictor/`
- `evaluator/`
- `plotter/`
- `interpreter/`

### Package responsibilities

- `trainer`: dataset split, gradient descent, model payload generation
- `predictor`: model loading/validation/versioning, mileage parsing, prediction safety checks
- `evaluator`: metrics, baseline comparison, split-aware evaluation, report serialization
- `plotter`: regression line visualization, residual diagnostics, export bundle (`metrics.json`, `summary.txt`)
- `interpreter`: converts JSON evaluation metrics into easy-to-read text for non-technical audiences

### Test suite

- `tests/test_trainer.py`
- `tests/test_predictor.py`
- `tests/test_plotter.py`
- `tests/test_interpreter.py`

## Python environment and `.venv/bin/python`

This project is designed to run inside a local virtual environment.

### Option A: activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install matplotlib
```

Then run commands with `python`:

```bash
python train.py
python predict.py --mileage 90000
```

### Option B: do not activate, call `.venv/bin/python` directly

This is explicit and very reliable in scripts/CI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install matplotlib
```

Run every command with:

```bash
.venv/bin/python train.py
.venv/bin/python predict.py --mileage 90000
.venv/bin/python evaluate.py --report report_artifacts/evaluation_report.json
```

Quick check of the interpreter actually used:

```bash
which python
python --version
.venv/bin/python --version
```

## Quick workflow with Makefile (recommended)

1. List available targets and variables:

```bash
make help
```

2. Create venv and install dependencies:

```bash
make deps
```

3. Run the full workflow:

```bash
make makeup
```

`make makeup` executes environment checks, lint, tests, training, evaluation, interpretation, plotting, and prediction.

### Common individual targets

```bash
make train
make predict MILEAGE=85000
make evaluate
make interpret
make plot
make test
```

### Useful parameter overrides

```bash
make train ITERATIONS=5000 LEARNING_RATE=0.05 TEST_RATIO=0.25
make plot PLOT_THEME=dark PLOT_FORMAT=svg
make evaluate EVAL_JSON=1 EVAL_REPORT=report_artifacts/evaluation_report.json
```

## Manual workflow without Makefile

Use these commands when you want full control.

1. Train:

```bash
.venv/bin/python train.py \
  --dataset data.csv \
  --model report_artifacts/model.json \
  --learning-rate 0.1 \
  --iterations 10000 \
  --test-ratio 0.2 \
  --seed 42 \
  --log-every 1000 \
  --early-stopping-patience 300 \
  --early-stopping-min-delta 1e-6
```

2. Predict:

```bash
.venv/bin/python predict.py --model report_artifacts/model.json --mileage 100000
.venv/bin/python predict.py --model report_artifacts/model.json --mileages 50000 80000 120000 --json
```

3. Evaluate:

```bash
.venv/bin/python evaluate.py \
  --dataset data.csv \
  --model report_artifacts/model.json \
  --test-ratio 0.2 \
  --seed 42 \
  --report report_artifacts/evaluation_report.json
```

4. Interpret evaluation report:

```bash
.venv/bin/python interpret.py \
  --report report_artifacts/evaluation_report.json \
  --output report_artifacts/interpretation_report.txt \
  --print
```

5. Generate diagnostic plot and report bundle:

```bash
.venv/bin/python plot.py \
  --dataset data.csv \
  --model report_artifacts/model.json \
  --output report_artifacts/regression_plot_make \
  --format png \
  --theme light \
  --report-dir report_artifacts
```

## Running tests

### All tests

```bash
make test
```

or:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

### Run tests by file

```bash
.venv/bin/python -m unittest tests.test_trainer -q
.venv/bin/python -m unittest tests.test_predictor -q
.venv/bin/python -m unittest tests.test_plotter -q
.venv/bin/python -m unittest tests.test_interpreter -q
```

## Generated artifacts

Outputs are centralized in `report_artifacts/`:

- `model.json`
- `evaluation_report.json`
- `interpretation_report.txt`
- `regression_plot_make.png` (or `.svg` / `.pdf`)
- `metrics.json`
- `summary.txt`

See `report_artifacts/README.md` for artifact details.

## Reusing this project as a developer

You can use this repository as a reusable mini-ML framework for small tabular regression tasks.

### Reuse pattern 1: keep CLI, replace dataset

- Keep module code unchanged
- Prepare a new `km,price` CSV file
- Run train/evaluate/plot/interpret with your new file paths

### Reuse pattern 2: call modules directly from Python

Example prediction integration:

```python
from pathlib import Path
import logging
from predictor.engine import load_model, predict
from predictor.model import ModelPolicy

logger = logging.getLogger("predict_integration")
model, _ = load_model(Path("report_artifacts/model.json"), ModelPolicy.STRICT, logger)
prices = predict(model, [45000.0, 90000.0, 135000.0])
print(prices)
```

Example evaluation integration:

```python
from pathlib import Path
import logging
from predictor.engine import load_model
from predictor.model import ModelPolicy
from evaluator.data import load_dataset
from evaluator.pipeline import evaluate

logger = logging.getLogger("eval_integration")
mileages, prices = load_dataset(Path("data.csv"))
model, _ = load_model(Path("report_artifacts/model.json"), ModelPolicy.STRICT, logger)
result = evaluate(mileages, prices, model, test_ratio=0.2, seed=42)
print(result.test.model.rmse)
```

### Reuse pattern 3: extend modules

Common extensions developers add:

- New features beyond mileage (multivariate regression)
- Better model persistence schema versions
- Additional metrics and dashboards
- CI jobs that run `make test` and `make makeup`

## Current limitations

- Single-feature linear regression only
- No regularization (L1/L2)
- No external optimized math libraries (by design)
- Educational scale, not optimized for very large datasets

## License

See `LICENSE`.
