# Decision Tree

The project implements a classification tree that recursively splits the training data using feature thresholds and information gain.

## Files

- [`../../DT.py`](../../DT.py): reusable `DecisionTree` implementation and its `node` class.
- [`../../Decision_Tree.ipynb`](../../Decision_Tree.ipynb): trains and evaluates the tree, draws the learned tree, plots 2D decision regions, and compares accuracy across tree depths.

## Implementation

`DecisionTree.fit(X, y)` grows the tree recursively. At each node it searches candidate feature/threshold splits and chooses the split with the greatest entropy-based information gain. It stops at a pure node, when `max_depth` is reached, or when there are fewer than `min_samples_split` samples. A leaf predicts the most common label among its samples. `predict(X)` traverses the learned tree for each row.

The default `n_features=None` uses all features. Candidate thresholds are unique observed feature values. Feature selection uses NumPy randomness when a subset is requested; the notebook does not set a random seed for this part.

## Run

Open `Decision_Tree.ipynb` in Jupyter and run its cells from top to bottom. It uses the built-in breast cancer dataset, a 70/30 train/test split with `random_state=42`, and scikit-learn for dataset loading and splitting. `DT.py` can also be imported by another Python script from this folder.

## Output

The notebook prints test accuracy, draws a tree visualization, plots decision regions using mean radius and mean texture only, and plots train/test accuracy for tree depths 1 through 10. The 2D decision-region plot is a separate tree trained on those two features; it is not the all-feature model.

## Dependencies

`numpy`, `matplotlib`, and `scikit-learn`.
