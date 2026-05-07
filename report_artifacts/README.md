# report_artifacts

Ce dossier contient les artefacts generes par `make plot` (ou `make makeup`) :

- `regression_plot.png` (ou `.svg` / `.pdf` selon `PLOT_FORMAT`)
- `metrics.json` : metriques du modele pour la visualisation
- `summary.txt` : resume rapide (plot, MSE, RMSE, MAE, outliers)

## Generer ces fichiers

```bash
make plot
```

Avec les valeurs par defaut du `Makefile`, la sortie est ecrite dans ce dossier.

Exemple avec theme sombre et SVG :

```bash
make plot PLOT_THEME=dark PLOT_FORMAT=svg
```
