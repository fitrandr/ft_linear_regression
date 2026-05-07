# ft_linear_regression

Introduction a la regression lineaire avec descente de gradient (42).

## Objectif du projet

Ce projet implemente un modele simple de Machine Learning pour predire le prix d'une voiture en fonction de son kilometrage.

Hypothese imposee par le sujet :

\[
\hat{y} = \theta_0 + \theta_1 \times x
\]

avec :
- \(x\) : kilometrage (km)
- \(\hat{y}\) : prix estime
- \(y\) : prix reel

## Ce qui est livre

### Partie obligatoire

- `train.py` : entraine le modele avec descente de gradient et sauvegarde `theta0`, `theta1` dans `model.json`.
- `predict.py` : lit `model.json` et predit le prix pour un kilometrage saisi.

### Partie bonus

- `evaluate.py` : calcule MAE, MSE, RMSE et R2.
- `plot.py` : affiche/sauvegarde le nuage de points et la droite de regression.
- `linear_regression_solution.ipynb` : notebook complet avec explications detaillees, formules Markdown/LaTeX et implementation pas a pas.

## Formules d'entrainement

Pour \(m\) exemples :

\[
\theta_0 := \theta_0 - \alpha \cdot \frac{1}{m}\sum_{i=0}^{m-1}(\hat{y}^{(i)} - y^{(i)})
\]

\[
\theta_1 := \theta_1 - \alpha \cdot \frac{1}{m}\sum_{i=0}^{m-1}(\hat{y}^{(i)} - y^{(i)})x^{(i)}
\]

Mise a jour **simultanee** de `theta0` et `theta1`.

## Structure

- `data.csv` : dataset (km, price)
- `train.py`
- `predict.py`
- `evaluate.py`
- `plot.py`
- `linear_regression_solution.ipynb`
- `model.json` (genere apres entrainement)

## Prerequis

- Python 3
- `matplotlib` uniquement pour les graphes (`plot.py` et certaines cellules du notebook)

Installation optionnelle :

```bash
pip install matplotlib
```

## Utilisation

### 1) Entrainer le modele

```bash
python3 train.py
```

Options utiles :

```bash
python3 train.py --dataset data.csv --model model.json --learning-rate 0.1 --iterations 10000
```

### 2) Predire un prix

Mode interactif :

```bash
python3 predict.py
```

Mode direct :

```bash
python3 predict.py --mileage 100000
```

### 3) Evaluer la precision (bonus)

```bash
python3 evaluate.py
```

### 4) Tracer la regression (bonus)

```bash
python3 plot.py
```

Genere `regression_plot.png`.

## Notebook detaille

Le fichier `linear_regression_solution.ipynb` contient :

1. Analyse du sujet.
2. Explications mathematiques detaillees.
3. Chargement et visualisation des donnees.
4. Entrainement par descente de gradient.
5. Conversion des thetas vers l'echelle reelle.
6. Evaluation (MAE, MSE, RMSE, R2).
7. Sauvegarde du modele et exemple de prediction.

## Notes

- Aucun outil de regression automatique (`numpy.polyfit`, `sklearn`, etc.) n'est utilise pour respecter le sujet.
- Si `model.json` n'existe pas, `predict.py` utilise \(\theta_0 = 0\) et \(\theta_1 = 0\), conformement a l'etat initial demande.
