# Logistic Regression Classifier

This notebook implements binary logistic regression using NumPy and gradient descent, then evaluates it on the Wisconsin breast cancer dataset.

## File and run

Open `Logistic_Regression.ipynb` in Jupyter and run cells in order. The notebook uses a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn (metrics and datasets).

## Mathematics
## Logistic regression: model and objective

### 1. The model

Each sample is a vector $x_i \in \mathbb{R}^d$ with $d$ features. The model works in two steps.

**Step 1: a linear score**, exactly as in linear regression:

$$
z_i = x_i^T w + b
$$

- $w \in \mathbb{R}^d$ is the **weight vector** and $b$ is the **intercept**.
- $z_i$ can be any real number, so it can't be used directly as a probability.

**Step 2: squash the score into a probability** with the sigmoid function:

$$
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} = P(y_i = 1 \mid x_i)
$$

Useful properties of the sigmoid:

- Its output is always strictly between 0 and 1.
- $\sigma(0) = 0.5$, and $\sigma(z) \to 1$ as $z \to +\infty$ and $\sigma(z) \to 0$ as $z \to -\infty$.
- It is symmetric: $\sigma(-z) = 1 - \sigma(z)$.
- Its derivative is $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$, which simplifies the gradient below.

For all $n$ samples at once, with $X \in \mathbb{R}^{n \times d}$:

$$
z = Xw + b\mathbf{1}, \qquad p = \sigma(z)
$$

where $\sigma$ is applied element-wise.

### 2. The loss: binary cross-entropy

For binary labels $y_i \in \{0, 1\}$, the model assigns probability $p_i$ to class 1 and $1 - p_i$ to class 0. The likelihood of one label is $p_i^{y_i}(1-p_i)^{1-y_i}$. Maximizing the likelihood of the whole dataset is the same as minimizing the **binary cross-entropy** (log loss):

$$
J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \Big]
$$

Only one of the two terms is active for each sample:

- If $y_i = 1$, the loss is $-\log p_i$. It is 0 when $p_i = 1$ and grows without bound as $p_i \to 0$.
- If $y_i = 0$, the loss is $-\log(1 - p_i)$. It is 0 when $p_i = 0$ and grows without bound as $p_i \to 1$.

So the loss punishes confident wrong predictions very heavily.

### 3. The gradients

Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, the sigmoid terms cancel and the gradient has a simple form:

$$
\nabla_w J = \frac{1}{n} X^T (p - y), \qquad
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)
$$

This is the same form as the linear regression gradient, with $\hat{y}$ replaced by $p$.

- For weight $w_j$, the gradient is the average of (error $\times$ feature value), where the error is $p_i - y_i$.
- For the intercept, it is the average error. If the model over-predicts class 1 on average, $b$ goes down.

### 4. Gradient descent

Starting from $w = 0$ and $b = 0$, repeat for a fixed number of iterations:

1. Compute scores $z = Xw + b$ and probabilities $p = \sigma(z)$.
2. Compute the gradients $\nabla_w J$ and $\partial J / \partial b$.
3. Update the parameters:

$$
w \leftarrow w - \eta \, \nabla_w J, \qquad
b \leftarrow b - \eta \, \frac{\partial J}{\partial b}
$$

where $\eta$ is the learning rate (`lr` in the code). This is **batch** gradient descent, because every update uses the full training set.

### 5. Prediction

The model predicts class 1 when the probability is above 0.5:

$$
\hat{y}_i =
\begin{cases}
1 & \text{if } p_i > 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

Because $\sigma(z) > 0.5$ exactly when $z > 0$, this is the same as checking the sign of the linear score. The **decision boundary** is therefore the hyperplane $x^T w + b = 0$, so logistic regression is a linear classifier. The 0.5 threshold can be changed if one kind of error is costlier than the other.

### 6. Practical notes

- **Numerical stability:** when $p_i$ is exactly 0 or 1, $\log(p_i)$ or $\log(1 - p_i)$ becomes $-\infty$. Clip probabilities (e.g. `np.clip(p, 1e-15, 1 - 1e-15)`) when computing the loss. The gradient is not affected.
- **Feature scaling:** standardize the features. It makes gradient descent converge much faster, and keeps $z$ from becoming large enough to saturate the sigmoid.
- **Convexity:** the loss $J$ is convex, so gradient descent with a suitable learning rate reaches the global minimum.
- **Perfectly separable data:** if the classes can be separated by a line, the weights grow without bound as training continues, since larger weights always reduce the loss. Adding L2 regularization (a penalty $\tfrac{\lambda}{2}\lVert w \rVert^2$ on the loss) prevents this.
- **Why not MSE?** With a sigmoid output, MSE is not convex and gives weak gradients when the model is confidently wrong. Cross-entropy avoids both problems.
- **Sanity check:** compare your accuracy and weights against scikit-learn's `LogisticRegression` (with `penalty=None`) on the same data.

## Output and interpretation

The notebook prints test accuracy and displays a confusion matrix. Rows are actual labels and columns are predicted labels. Diagonal cells are correct classifications; off-diagonal cells show errors (false positives and false negatives). Dataset class names label the axes.

## Limitations

- The 30 input features are used without scaling. Feature magnitudes can affect optimization and the learning rate's suitability.
- The notebook does not retain training loss, return probabilities through a separate method, or apply regularization.
- The direct sigmoid formula may overflow for extreme scores; a numerically stable implementation is not used.
- Accuracy alone can hide which class is misclassified; inspect the confusion matrix as well.
