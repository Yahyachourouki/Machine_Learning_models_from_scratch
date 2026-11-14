# Linear Support Vector Machine (SVM)

This notebook implements a linear binary classifier with a hinge-loss-style update, evaluates it on the breast cancer test split, and creates a PCA visualization.

## File and run

Open `SVM.ipynb` in Jupyter and run cells in order. It uses a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn (datasets, metrics, scaling, PCA).

## Mathematics
## Linear SVM: model and objective

### 1. The model

Labels are encoded as $y_i \in \{-1, +1\}$. In the code, any label $\le 0$ becomes $-1$ and everything else becomes $+1$.

The classifier computes a linear score

$$
f(x) = w^T x + b
$$

- $w \in \mathbb{R}^d$ is the **weight vector**, perpendicular to the separating hyperplane.
- $b$ is the **bias** (offset), which shifts the hyperplane away from the origin.

The sign of $f(x)$ tells you which side of the hyperplane a point lies on, so the prediction is $\hat{y} = \operatorname{sign}(f(x))$.

### 2. Geometry: boundary and margin

- **Decision boundary:** $f(x) = 0$.
- **Margin lines:** $f(x) = +1$ and $f(x) = -1$, one on each side of the boundary.
- **Distance of a point to the boundary:** $|f(x)| / \lVert w \rVert_2$.
- **Margin width** (distance between the two margin lines): $2 / \lVert w \rVert_2$.

A point is **correctly classified** if $y_i f(x_i) > 0$. The SVM goal is stronger: $y_i f(x_i) \ge 1$, meaning the point is correctly classified **and** outside the margin.

Maximizing the margin is the same as making $\lVert w \rVert_2$ small. This is why the objective below contains a weight penalty.

### 3. The hinge loss

For one sample, the hinge loss is

$$
\ell_i = \max\big(0,\; 1 - y_i f(x_i)\big)
$$

| Situation | $y_i f(x_i)$ | Hinge loss |
|---|---|---|
| Correct, outside the margin | $\ge 1$ | $0$ |
| Correct, but inside the margin | between $0$ and $1$ | between $0$ and $1$ |
| Misclassified | $< 0$ | greater than $1$, growing linearly |

For example, with $y_i = +1$: $f = 2$ gives loss 0, $f = 0.4$ gives loss 0.6, and $f = -0.5$ gives loss 1.5.

Points that are safely correct contribute nothing, so only points **on or inside the margin, or misclassified,** affect the solution.

### 4. The soft-margin objective

A common form of the objective is

$$
\min_{w,\, b}\;\; \lambda \lVert w \rVert_2^2 \;+\; \frac{1}{n} \sum_{i=1}^{n} \max\big(0,\; 1 - y_i (w^T x_i + b)\big)
$$

- The **first term** is the regularizer. It keeps $w$ small, which gives a **wide margin**.
- The **second term** is the average hinge loss. It penalizes margin violations and misclassifications.
- $\lambda$ balances the two. A **large $\lambda$** gives a wider margin and tolerates more violations (simpler model). A **small $\lambda$** fits the training data more tightly and risks overfitting.

This is equivalent to the textbook form $\tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i$ with $C = \dfrac{1}{2 \lambda n}$.

### 5. Gradient (subgradient) updates

The hinge loss has a kink at $y_i f(x_i) = 1$, so it is not differentiable everywhere. Gradient descent uses a **subgradient**. For a single sample $(x_i, y_i)$:

**If $y_i f(x_i) \ge 1$** (outside the margin), only the regularizer contributes:

$$
\nabla_w = 2 \lambda w, \qquad \frac{\partial}{\partial b} = 0
$$

**If $y_i f(x_i) < 1$** (violating the margin), the hinge term also contributes:

$$
\nabla_w = 2 \lambda w - y_i x_i, \qquad \frac{\partial}{\partial b} = -y_i
$$

The extra $-y_i x_i$ is the **class-weighted feature term**. It pushes $w$ toward $x_i$ for positive samples and away from it for negative samples, which moves the margin violator toward the correct side.

Each step then updates the parameters with learning rate $\eta$:

$$
w \leftarrow w - \eta \, \nabla_w, \qquad b \leftarrow b - \eta \, \frac{\partial}{\partial b}
$$

The notebook processes **one sample at a time** (stochastic gradient descent) and loops over the full dataset for `n_iterations` passes.

### 6. Support vectors

Points with $y_i f(x_i) \le 1$ (**on or inside a margin**, including misclassified points) are commonly called **support vectors**. They are the only points that produce a non-zero hinge update, so they determine the final boundary. Moving or removing a point outside the margin does not change the solution.

### 7. Known issues in the implementation

The code follows the standard objective **in spirit, but not exactly**, because of these issues:

1. **Bias sign mismatch.** `fit` uses $f(x) = w^T x - b$, but `predict` uses $w^T x + b$. They need the same convention.
2. **Learning rate applied twice.** `dw` and `db` already include `lr`, and the update multiplies by `lr` again, so the effective step size is $\eta^2$ instead of $\eta$. Training is much slower than intended.
3. **Label mismatch in evaluation.** `predict` returns $\{-1, +1\}$, but test labels are $\{0, 1\}$. Convert them before computing accuracy.
4. **No feature scaling.** Linear SVMs are sensitive to feature scales, so standardize the features first.

So the formulas above describe the standard objective, **not an exact guarantee of what the code optimizes**, until these are fixed.

### 8. Practical notes

- **Convexity:** the objective is convex, so there are no local minima. Convergence still depends on the learning rate and iteration count.
- **Linear only:** this model finds a straight hyperplane. Non-linear boundaries need kernels (e.g. RBF), which this implementation doesn't include.
- **Visualization:** for data with more than 2 features, project to 2D with PCA before plotting the boundary, margins and support vectors.
- **Sanity check:** compare against scikit-learn's `LinearSVC` or `SGDClassifier(loss="hinge")` on the same standardized data. The accuracy should be close.

## Evaluation and plot

The dataset labels 0/1 are mapped internally to -1/+1. Predictions are mapped back before the later confusion matrix and accuracy calculation. The visualization standardizes training features, projects them to two components with PCA, trains a separate 2D SVM, and plots classes, a boundary, margins, and an approximate support-vector mask. This plot is not the 30-feature model's boundary.

## Important code caveats

- **Bias mismatch:** `fit` uses `dot(x_i, weights) - bias` for its margin condition, while `predict` uses `dot(X, weights) + bias`. The PCA plot also uses subtraction. These are inconsistent decision functions, so predictions and the displayed boundary may disagree.
- **Learning-rate scaling:** each branch multiplies `dw` by `lr`, and the update multiplies it by `lr` again. This makes the weight step scale as `lr²`, not the usual single learning-rate factor. The bias update is also scaled inside the branch and again during the update.
- The notebook prints an initial accuracy comparing -1/+1 predictions directly with 0/1 labels; that number is not meaningful. Use the later accuracy after converting labels.
- `np.sign` can return 0 for an exact zero score; the conversion maps values other than -1 to 1.
- The support-vector mask is approximate, based on the current score and tolerance; it is not extracted from a constrained optimization solver.

Fix these inconsistencies before treating the metric or plot as a validated SVM result.
