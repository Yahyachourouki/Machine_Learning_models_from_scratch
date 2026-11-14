# Random Forest Classifier

The notebook combines many decision trees. Each tree sees a bootstrap sample of the training rows and considers a random subset of features at each split. The forest predicts by majority vote.

## Files and run

Open `Random_Forest.ipynb` in Jupyter and run cells in order. It imports `DecisionTree` from the adjacent `DT.py`, so keep both files together. The example uses the breast cancer dataset with a 70/30 split (`random_state=42`). Requirements: NumPy and scikit-learn.

## Mathematics
## Random forest: model and prediction

A random forest is an **ensemble** of $B$ decision trees. Each tree is trained on a different random view of the data, and the trees vote on the final prediction. Averaging many different trees gives a more stable model than any single tree.

### 1. Training: building the forest

For each tree $b = 1, \ldots, B$:

1. **Bootstrap sampling.** Draw $n$ training rows **with replacement** from the $n$ available rows. Some rows appear several times and others not at all.
2. **Fit a tree** on that bootstrap sample, using the information-gain criterion described in the [Decision Tree README](../decision_tree/README.md).
3. **Random feature subsets.** At each split, the tree searches only $m$ randomly chosen candidate features instead of all $d$. The default here is

$$
m \approx \sqrt{d}
$$

A new random subset is drawn at every split, not once per tree.

The trees are grown independently of each other, so the process could be run in parallel.

### 2. Why the bootstrap sample is different each time

When drawing $n$ rows with replacement, the chance that a particular row is **never** picked is

$$
\left(1 - \frac{1}{n}\right)^n \;\xrightarrow{\,n \to \infty\,}\; e^{-1} \approx 0.368
$$

So each tree sees about **63.2%** of the unique training rows. The remaining ~36.8% are called that tree's **out-of-bag (OOB)** samples.

### 3. Prediction: majority vote

Each tree predicts a class $h_b(x)$. The forest predicts the most common class among them:

$$
\hat{y}(x) = \operatorname{mode}\big(h_1(x),\, h_2(x),\, \ldots,\, h_B(x)\big)
$$

Example: with $B = 5$ trees voting $(1, 0, 1, 1, 0)$, the forest predicts class 1 (3 votes against 2).

In the code, this is `np.bincount(votes).argmax()` for each sample. If two classes tie, `argmax` returns the **smaller** class label. Using an odd number of trees avoids ties in binary problems.

### 4. Why it works

Each tree on its own is a **low-bias, high-variance** model: a deep tree fits its training data very closely, but a small change in the data can produce a very different tree. Averaging reduces this variance. If the trees each have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of their average is

$$
\rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2
$$

This explains the two sources of randomness:

- **More trees** ($B$ larger) shrinks the second term toward 0.
- **Bootstrapping and random feature subsets** make the trees less alike (lower $\rho$), which shrinks the first term. Without the feature subsets, strong features would dominate every tree's early splits and the trees would be highly correlated.

Majority voting can reduce the variance of a single tree, but it does not guarantee better performance in every case. If the trees are highly correlated or all biased in the same way, the gain is small.

### 5. Hyperparameters

| Parameter | Meaning | Effect |
|---|---|---|
| `n_estimators` ($B$) | Number of trees | More trees give more stable results, with diminishing returns and a longer training time. They rarely cause overfitting. |
| `max_depth` | Maximum depth of each tree | Deeper trees have lower bias but higher variance. |
| `min_samples_split` | Minimum samples needed to split a node | Larger values give simpler trees. |
| `n_feats` ($m$) | Candidate features per split | Smaller values give less correlated trees but weaker individual trees. |

### 6. Practical notes

- **Speed:** training cost grows linearly with $B$, and the threshold search over every unique feature value is slow in pure Python. Use a small `n_estimators` (e.g. 10) and a limited `max_depth` while testing.
- **Out-of-bag evaluation:** predict each training row using only the trees that did **not** see it in their bootstrap sample. This gives a validation estimate without a separate hold-out set.
- **Reproducibility:** bootstrapping and feature sampling are random, so set `np.random.seed(...)` to get repeatable results.
- **No feature scaling needed:** trees split on thresholds, so standardizing features does not change the result.
- **Sanity check:** compare your accuracy against scikit-learn's `RandomForestClassifier` with similar settings. Results should be close but not identical, because of the randomness.
## Output

The notebook fits 10 trees with maximum depth 5 and prints test accuracy. It does not currently plot the forest or show a confusion matrix.

## Implementation notes

- Set `n_feats` to override the default feature count; otherwise it uses `max(1, int(sqrt(d)))`.
- Bootstrap sampling and tree feature selection are not seeded, so results can change from run to run even though the train/test split is fixed.
- Majority voting uses `np.bincount`; class labels therefore need to be nonnegative integers.
- The implementation has no out-of-bag score, class probabilities, or parallel fitting. It is an educational version, not a full-featured forest library.
