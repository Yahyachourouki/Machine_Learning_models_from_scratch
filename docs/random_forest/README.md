# Random Forest Classifier

The notebook combines many decision trees. Each tree sees a bootstrap sample of the training rows and considers a random subset of features at each split. The forest predicts by majority vote.

## Files and run

Open `Random_Forest.ipynb` in Jupyter and run cells in order. It imports `DecisionTree` from the adjacent `DT.py`, so keep both files together. The example uses the breast cancer dataset with a 70/30 split (`random_state=42`). Requirements: NumPy and scikit-learn.

## Mathematics

For each tree \(b=1,\ldots,B\), bootstrap sampling draws \(n\) training rows with replacement. The tree is fit on that sample. At each split it considers \(m\) candidate features (the default here is approximately \(\sqrt d\) for \(d\) input features) and uses the information-gain criterion described in the [Decision Tree README](../decision_tree/README.md).

Each tree predicts class \(h_b(x)\). The forest prediction is the most common class:

\[
\hat y(x)=\operatorname{mode}(h_1(x),h_2(x),\ldots,h_B(x)).
\]

Bootstrapping and random feature subsets make trees less alike. Majority voting can reduce the variance of a single tree, though it does not guarantee better performance in every case.

## Output

The notebook fits 10 trees with maximum depth 5 and prints test accuracy. It does not currently plot the forest or show a confusion matrix.

## Implementation notes

- Set `n_feats` to override the default feature count; otherwise it uses `max(1, int(sqrt(d)))`.
- Bootstrap sampling and tree feature selection are not seeded, so results can change from run to run even though the train/test split is fixed.
- Majority voting uses `np.bincount`; class labels therefore need to be nonnegative integers.
- The implementation has no out-of-bag score, class probabilities, or parallel fitting. It is an educational version, not a full-featured forest library.
