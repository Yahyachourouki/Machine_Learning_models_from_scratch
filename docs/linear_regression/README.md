# Linear Regression

This notebook implements a linear model with batch gradient descent, plots the learned line, evaluates test predictions, and compares its parameters with scikit-learn.

## File and run

Open `linear_regression.ipynb` in Jupyter and run cells in order. It generates 100 synthetic samples with one feature using `make_regression` (`noise=20`, `random_state=4`), then splits 80/20 (`random_state=42`). Requirements: NumPy, Matplotlib, scikit-learn.

## Model and objective

For row \(x_i\in\mathbb{R}^d\), prediction is

\[
\hat y_i=x_i^T w+b,
\]

where \(w\) is the feature-weight vector and \(b\) is the intercept. Training minimizes mean squared error:

\[
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(x_i^Tw+b-y_i)^2.
\]

The code uses the gradient of half the MSE (equivalently, it omits a factor of 2 from the MSE gradient):

\[
\nabla_wJ=\frac{1}{n}X^T(\hat y-y),\qquad
\frac{\partial J}{\partial b}=\frac{1}{n}\sum_i(\hat y_i-y_i).
\]

Each iteration subtracts `lr` times these gradients from the parameters. It is batch gradient descent because every update uses the complete training set.

## Output and interpretation

The notebook plots training points and the fitted line, plots the `costs` list, prints test MSE and R², and compares weights and bias with scikit-learn's coefficient and intercept. MSE is the average squared prediction error. R² is \(1-SS_{res}/SS_{tot}\); 1 is a perfect fit, 0 is the mean baseline, and a negative value is worse than that baseline on the evaluated data.

## Cost-history caveat

The first loop in `fit` appends the initial MSE `n_iterations` times without updating parameters. A second loop then appends MSE before each update. Thus `costs` has `2*n_iterations` values; the first half is a repeated initial value, and the second half records pre-update costs. The plotted x-axis is a list index, not a clean iteration history. This notebook's plot should be interpreted with that in mind.

The example has one feature, though the class uses vector weights and accepts multiple features. It has no regularization, feature scaling, or early stopping.

