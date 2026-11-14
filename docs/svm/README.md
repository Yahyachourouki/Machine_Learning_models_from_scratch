# Linear Support Vector Machine (SVM)

This notebook implements a linear binary classifier with a hinge-loss-style update, evaluates it on the breast cancer test split, and creates a PCA visualization.

## File and run

Open `SVM.ipynb` in Jupyter and run cells in order. It uses a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn (datasets, metrics, scaling, PCA).

## Mathematics

The classifier uses labels \(y_i\in\{-1,+1\}\) and a linear score \(f(x)=w^Tx+b\). The predicted sign indicates the side of the separating hyperplane. A point is correctly classified if \(y_if(x_i)>0\); the unit-margin goal is \(y_if(x_i)\ge1\).

A common soft-margin objective is

\[
\min_{w,b}\;\lambda\lVert w\rVert_2^2+\frac1n\sum_i\max(0,1-y_i(w^Tx_i+b)).
\]

The hinge loss \(\max(0,1-y_if(x_i))\) is zero outside the margin and positive for points inside it or misclassified. The decision boundary is \(f(x)=0\); margin lines are \(f(x)=\pm1\). Points on or inside a margin are commonly called support vectors.

The notebook's intended update regularizes weights and adds a class-weighted feature term for samples violating the margin. However, the implementation has issues described below, so this formula explains the standard objective, not an exact guarantee of what the code optimizes.

## Evaluation and plot

The dataset labels 0/1 are mapped internally to -1/+1. Predictions are mapped back before the later confusion matrix and accuracy calculation. The visualization standardizes training features, projects them to two components with PCA, trains a separate 2D SVM, and plots classes, a boundary, margins, and an approximate support-vector mask. This plot is not the 30-feature model's boundary.

## Important code caveats

- **Bias mismatch:** `fit` uses `dot(x_i, weights) - bias` for its margin condition, while `predict` uses `dot(X, weights) + bias`. The PCA plot also uses subtraction. These are inconsistent decision functions, so predictions and the displayed boundary may disagree.
- **Learning-rate scaling:** each branch multiplies `dw` by `lr`, and the update multiplies it by `lr` again. This makes the weight step scale as `lr²`, not the usual single learning-rate factor. The bias update is also scaled inside the branch and again during the update.
- The notebook prints an initial accuracy comparing -1/+1 predictions directly with 0/1 labels; that number is not meaningful. Use the later accuracy after converting labels.
- `np.sign` can return 0 for an exact zero score; the conversion maps values other than -1 to 1.
- The support-vector mask is approximate, based on the current score and tolerance; it is not extracted from a constrained optimization solver.

Fix these inconsistencies before treating the metric or plot as a validated SVM result.
