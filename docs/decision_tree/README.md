# Decision Tree Classifier

The notebook builds a classifier by recursively splitting the training set. An internal node tests whether one feature is at most a threshold; a leaf predicts the most common class among the training rows that reach it.

## File and run

Open `Decision_Tree.ipynb` in Jupyter and run cells in order. It loads scikit-learn's built-in Wisconsin breast cancer dataset and makes a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn. The reusable `DecisionTree` implementation is also in `../random_forest/DT.py`.

## Mathematics

## How the tree chooses a split

### 1. Entropy: how mixed is a node?

Let $S$ be the set of samples in a node, and let $p_k$ be the fraction of those samples that belong to class $k$. The entropy is

$$
H(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

(with the convention $0 \log_2 0 = 0$).

- $H(S) = 0$: the node is **pure** (all samples have the same class).
- $H(S) = 1$: a binary node is split **50/50**, the most mixed case.

### 2. Information gain: how good is a split?

A candidate split uses a feature $j$ and a threshold $t$ to divide $S$ into two groups:

$$
S_L = \{\, i : x_{ij} \le t \,\}, \qquad S_R = \{\, i : x_{ij} > t \,\}
$$

Here $x_{ij}$ is the value of feature $j$ for sample $i$. The information gain is how much the entropy drops after the split, with each child weighted by its share of the samples:

$$
IG(S, j, t) = H(S) - \frac{|S_L|}{|S|}\, H(S_L) - \frac{|S_R|}{|S|}\, H(S_R)
$$

A higher gain means the split separates the classes better.

### 3. Growing the tree

1. Try every unique value of each candidate feature as a threshold $t$.
2. Pick the pair $(j, t)$ with the **highest** information gain.
3. Split the node and repeat on the left and right children.

A node becomes a **leaf** when any of these is true:

- it is pure (only one class left),
- it reaches `max_depth`,
- it has fewer than `min_samples_split` samples.

A leaf predicts the **majority class** of the samples in it.

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
