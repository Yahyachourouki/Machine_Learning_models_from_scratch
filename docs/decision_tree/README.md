# Decision Tree Classifier

The notebook builds a classifier by recursively splitting the training set. An internal node tests whether one feature is at most a threshold; a leaf predicts the most common class among the training rows that reach it.

## File and run

Open `Decision_Tree.ipynb` in Jupyter and run cells in order. It loads scikit-learn's built-in Wisconsin breast cancer dataset and makes a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn. The reusable `DecisionTree` implementation is also in `../random_forest/DT.py`.

## Mathematics

For a node with class fractions \(p_1,\ldots,p_K\), entropy measures class mixture:

\[
H(S)=-\sum_{k=1}^{K}p_k\log_2(p_k).
\]

A candidate split on feature \(j\) at threshold \(t\) creates \(S_L=\{i:x_{ij}\le t\}\) and \(S_R=\{i:x_{ij}>t\}\). Information gain is the reduction in entropy:

\[
IG(S,j,t)=H(S)-\frac{|S_L|}{|S|}H(S_L)-\frac{|S_R|}{|S|}H(S_R).
\]

The code chooses the feature-threshold pair with maximum gain, then repeats the process in each child. It stops when the node is pure, reaches `max_depth`, or has fewer rows than `min_samples_split`. A leaf uses the majority label.

## Notebook outputs

It prints test accuracy, draws the trained tree, plots 2D decision regions, and plots train/test accuracy for depths 1–10. The first tree uses all 30 features with depth 5. The 2D plot trains a separate depth-4 tree using mean radius and mean texture only, so it is a visualization of a different model. The tree drawing is also truncated at the requested display depth.

## Implementation notes

- If `n_features` is omitted, all features are considered at each split. Otherwise the code randomly chooses that many features at each node.
- The feature-subset selection is not seeded, so tree results can differ between runs.
- Entropy uses `np.bincount`, so labels must be nonnegative integers (as in the notebook's 0/1 labels).
- Candidate thresholds are the observed unique values; the code does not search midpoints between adjacent values.
- The implementation is a teaching example and does not support arbitrary categorical/string features or missing values.

## Dependencies

NumPy, Matplotlib, scikit-learn.
