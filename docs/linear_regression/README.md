# Linear Regression

This notebook implements a linear model with batch gradient descent, plots the learned line, evaluates test predictions, and compares its parameters with scikit-learn.

## File and run

Open `linear_regression.ipynb` in Jupyter and run cells in order. It generates 100 synthetic samples with one feature using `make_regression` (`noise=20`, `random_state=4`), then splits 80/20 (`random_state=42`). Requirements: NumPy, Matplotlib, scikit-learn.

## model and objective

### 1. The model

Each sample is a vector $x_i \in \mathbb{R}^d$ with $d$ features. The model predicts

$$
\hat{y}_i = x_i^T w + b
$$

- $w \in \mathbb{R}^d$ is the **weight vector**, one weight per feature.
- $b \in \mathbb{R}$ is the **intercept** (bias), the prediction when all features are 0.

For all $n$ samples at once, stack the rows into a matrix $X \in \mathbb{R}^{n \times d}$ and the targets into a vector $y \in \mathbb{R}^n$:

$$
\hat{y} = Xw + b\mathbf{1}
$$

where $\mathbf{1}$ is a vector of ones. This matrix form is what the NumPy code uses.

### 2. The loss

Training measures error with the **mean squared error (MSE)**:

$$
\text{MSE}(w,b) = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2
$$

Squaring makes every error positive and penalizes large mistakes more than small ones.

The code minimizes **half** the MSE:

$$
J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}\left(x_i^T w + b - y_i\right)^2
$$

The factor $\tfrac{1}{2}$ cancels the 2 that appears when differentiating the square. Both losses have the same minimizer. Only the scale of the gradient, and therefore the effective learning rate, changes.

### 3. The gradients

Let $e_i = \hat{y}_i - y_i$ be the error for sample $i$. Differentiating $J$ gives

$$
\nabla_w J = \frac{1}{n} X^T(\hat{y} - y), \qquad
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

How to read them:

- For weight $w_j$, the gradient is the average of (error $\times$ feature value). If a feature is large exactly when the model over-predicts, its weight is pushed down.
- For the intercept, it is just the average error: if the model over-predicts on average, $b$ goes down.

### 4. Gradient descent

Starting from $w = 0$ and $b = 0$, repeat for a fixed number of iterations:

1. Compute predictions $\hat{y} = Xw + b$.
2. Compute the gradients $\nabla_w J$ and $\partial J / \partial b$.
3. Update the parameters:

$$
w \leftarrow w - \eta \, \nabla_w J, \qquad
b \leftarrow b - \eta \, \frac{\partial J}{\partial b}
$$

where $\eta$ is the learning rate (`lr` in the code).

This is **batch** gradient descent, because every update uses the complete training set. (Stochastic gradient descent would use one sample per update, and mini-batch would use a small subset.)

### 5. Practical notes

- **Learning rate:** if `lr` is too large, the loss oscillates or diverges. If it is too small, training is slow. A good sanity check is that the loss should decrease every iteration.
- **Feature scaling:** features on very different scales make the loss surface elongated, so gradient descent converges slowly. Standardizing the features (zero mean, unit variance) usually helps a lot.
- **Convexity:** $J$ is convex, so gradient descent with a suitable learning rate converges to the global minimum. There are no local minima to get stuck in.
- **Closed-form check:** linear regression also has an exact solution, the normal equation $w = (X^T X)^{-1} X^T y$ (with a column of ones appended to $X$ for the intercept). Comparing your result with it, or with scikit-learn's `LinearRegression`, is a good way to test your implementation.
## Output and interpretation

The notebook plots training points and the fitted line, plots the `costs` list, prints test MSE and R², and compares weights and bias with scikit-learn's coefficient and intercept. MSE is the average squared prediction error. R² is \(1-SS_{res}/SS_{tot}\); 1 is a perfect fit, 0 is the mean baseline, and a negative value is worse than that baseline on the evaluated data.

## Cost-history caveat

The first loop in `fit` appends the initial MSE `n_iterations` times without updating parameters. A second loop then appends MSE before each update. Thus `costs` has `2*n_iterations` values; the first half is a repeated initial value, and the second half records pre-update costs. The plotted x-axis is a list index, not a clean iteration history. This notebook's plot should be interpreted with that in mind.

The example has one feature, though the class uses vector weights and accepts multiple features. It has no regularization, feature scaling, or early stopping.

