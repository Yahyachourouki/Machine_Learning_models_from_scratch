# Machine Learning from Scratch

Implementations and demonstrations of five machine-learning models using Python and NumPy. The model code is written for learning purposes; scikit-learn is used in the notebooks to load datasets, split data, and (for linear regression) compare results.

## Models

| Model | Implementation and example | Documentation |
|---|---|---|
| Decision Tree | [`DT.py`](DT.py), [`Decision_Tree.ipynb`](Decision_Tree.ipynb) | [Decision tree README](docs/decision_tree/README.md) |
| Linear Regression | [`linear_regression.ipynb`](linear_regression.ipynb) | [Linear regression README](docs/linear_regression/README.md) |
| Logistic Regression | [`Logistic_Regression.ipynb`](Logistic_Regression.ipynb) | [Logistic regression README](docs/logistic_regression/README.md) |
| Random Forest | [`Random_Forest.ipynb`](Random_Forest.ipynb) | [Random forest README](docs/random_forest/README.md) |
| Support Vector Machine (SVM) | [`SVM.ipynb`](SVM.ipynb) | [SVM README](docs/svm/README.md) |

## Repository layout

```text
ch4/
├── DT.py
├── Decision_Tree.ipynb
├── Logistic_Regression.ipynb
├── Random_Forest.ipynb
├── SVM.ipynb
├── linear_regression.ipynb
└── docs/
    ├── decision_tree/README.md
    ├── linear_regression/README.md
    ├── logistic_regression/README.md
    ├── random_forest/README.md
    └── svm/README.md
```

The project currently keeps the model notebooks in the repository root; the `docs/` subfolders contain documentation only.

## Requirements

The notebooks import Python packages `numpy`, `matplotlib`, and `scikit-learn`. Install them in your Python environment if needed:

```bash
python -m pip install numpy matplotlib scikit-learn jupyter
```

## Run the notebooks

Open a terminal in this folder, start Jupyter, and open a notebook:

```bash
jupyter lab
```

Run notebook cells from top to bottom. The classification examples use scikit-learn's built-in Wisconsin breast cancer dataset. The linear regression notebook creates synthetic regression data with `sklearn.datasets.make_regression`; no external dataset files are required.

The Random Forest notebook imports `DecisionTree` from `DT.py`, so keep both files in the same folder and start Jupyter with this folder as the working directory.

## Notes

- These are educational implementations and are not intended as drop-in replacements for production ML libraries.
- Random choices are made in the tree and forest implementations; results can vary between runs because the code does not set a NumPy random seed for those choices.
- The notebooks contain different examples and plotting choices; see each model README for details and implementation caveats.
