# Machine Learning from Scratch

A beginner-oriented collection of five machine-learning implementations. NumPy contains the model calculations; scikit-learn supplies datasets, train/test splitting, metrics, PCA, scaling, and a linear-regression comparison.

## Models

| Model | Notebook / code | Detailed guide |
|---|---|---|
| Decision Tree | [Decision_Tree.ipynb](docs/decision_tree/Decision_Tree.ipynb) | [Math and implementation](docs/decision_tree/README.md) |
| Linear Regression | [linear_regression.ipynb](docs/linear_regression/linear_regression.ipynb) | [Math and implementation](docs/linear_regression/README.md) |
| Logistic Regression | [Logistic_Regression.ipynb](docs/logistic_regression/Logistic_Regression.ipynb) | [Math and implementation](docs/logistic_regression/README.md) |
| Random Forest | [Random_Forest.ipynb](docs/random_forest/Random_Forest.ipynb) | [Math and implementation](docs/random_forest/README.md) |
| Linear SVM | [SVM.ipynb](docs/svm/SVM.ipynb) | [Math and implementation](docs/svm/README.md) |

The Random Forest notebook imports its tree implementation from [DT.py](docs/random_forest/DT.py).

## Setup and run

From the project root, install dependencies and launch Jupyter:

```bash
python -m pip install numpy matplotlib scikit-learn jupyter
jupyter lab
```

Open a notebook and run cells in order. Start Jupyter from this root so `Random_Forest.ipynb` can import `DT.py` from its folder. No external dataset files are needed.

## Data used

- Decision Tree, Logistic Regression, Random Forest, and SVM use scikit-learn's built-in Wisconsin breast cancer dataset with a 70/30 split (`random_state=42`).
- Linear Regression generates 100 synthetic one-feature samples (`noise=20`, `random_state=4`) and uses an 80/20 split (`random_state=42`).

## Project scope

These are learning implementations, not production estimators. Review each model guide for formulas, output interpretation, and code-specific caveats. Tree feature selection and Random Forest bootstrap sampling are unseeded, so those results may change across runs.
