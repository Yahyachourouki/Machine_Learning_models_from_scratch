# Random Forest

This notebook implements a classifier that combines predictions from multiple decision trees.

## Files

- [`../../Random_Forest.ipynb`](../../Random_Forest.ipynb): defines `RandomForest` and runs the example.
- [`../../DT.py`](../../DT.py): supplies the `DecisionTree` class imported by the notebook.

## Implementation

`RandomForest.fit(X, y)` creates bootstrap samples by sampling training rows with replacement. It trains one decision tree per sample and uses a default feature count of approximately the square root of the number of input features. `predict(X)` collects each tree's predictions and returns the most frequent class for each row.

## Run

Open `Random_Forest.ipynb` in Jupyter and run cells from top to bottom, with `DT.py` in the same project folder. The example uses the built-in breast cancer dataset, a 70/30 train/test split with `random_state=42`, and 10 trees with `max_depth=5`.

## Output

The notebook prints test accuracy. It does not currently create a plot or display a confusion matrix.

## Implementation note

The tree and bootstrap sampling use NumPy randomness without a fixed seed, so repeated runs can produce different forests and accuracy values. The forest uses the custom tree implementation rather than scikit-learn's random forest.

## Dependencies

`numpy` and `scikit-learn`. `DT.py` is a local project dependency.
