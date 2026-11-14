# Support Vector Machine (SVM)

This notebook contains a linear binary SVM implementation trained using a hinge-loss-style update, along with accuracy, confusion-matrix, and 2D visualization examples.

## File

- [`../../SVM.ipynb`](../../SVM.ipynb)

## Implementation

`SVM.fit(X, y)` maps labels less than or equal to zero to -1 and other labels to +1, then updates weights and bias over the training samples for each iteration. `predict(X)` returns the sign of the linear score, producing -1 or +1 predictions.

## Run

Open the notebook in Jupyter and run cells from top to bottom. It uses scikit-learn's breast cancer dataset and a 70/30 train/test split with `random_state=42`.

## Output

The notebook prints accuracy and displays a confusion matrix after converting SVM predictions from -1/+1 to the dataset's 0/1 labels. It also scales the training features, projects them to two dimensions with PCA, trains a separate SVM in that projected space, and plots its linear boundary, margins, and approximate support vectors. The PCA plot is a visualization model and is not the original 30-feature model.

## Implementation caveat

The fitting condition uses `dot(x_i, weights) - bias`, while `predict` uses `dot(X, weights) + bias`. The plotted score uses the subtraction convention. These inconsistent bias signs mean the plotted boundary and predictions may not agree. Also, `np.sign` can return 0 for an exactly zero score. Review these behaviors before interpreting the plot or accuracy as a validated SVM result.

## Dependencies

`numpy`, `matplotlib`, and `scikit-learn` (including its `decomposition` and `preprocessing` modules).
