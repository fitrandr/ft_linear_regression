# report_artifacts

This directory stores generated outputs produced by the training, evaluation, interpretation, and plotting pipelines.

## Expected artifacts

- `model.json`: trained model payload and training metadata
- `evaluation_report.json`: full evaluation report (full/train/test metrics and baseline comparison)
- `interpretation_report.txt`: plain-language interpretation of model performance
- `regression_plot_make.png` (or `.svg` / `.pdf`): regression + residual diagnostics chart
- `metrics.json`: plot-oriented metrics bundle
- `summary.txt`: short summary of key metrics

## Generate artifacts with Makefile

Run the complete pipeline:

```bash
make makeup
```

Run only artifact generation:

```bash
make artifacts
```

Generate plot in dark SVG mode:

```bash
make plot PLOT_THEME=dark PLOT_FORMAT=svg
```

## Generate artifacts manually

```bash
python train.py --dataset data.csv --model report_artifacts/model.json
python evaluate.py --dataset data.csv --model report_artifacts/model.json --report report_artifacts/evaluation_report.json
python interpret.py --report report_artifacts/evaluation_report.json --output report_artifacts/interpretation_report.txt
python plot.py --dataset data.csv --model report_artifacts/model.json --output report_artifacts/regression_plot_make --report-dir report_artifacts
```

## Cleanup policy

The `make fclean` target deletes generated files in this directory while preserving this `README.md` file.
