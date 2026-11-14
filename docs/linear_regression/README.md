# Linear Regression

This notebook implements one-feature linear regression using gradient descent and compares the learned coefficients with scikit-learn's linear regression.

## File

- [`../../linear_regression.ipynb`](../../linear_regression.ipynb)

## Implementation

The `LinearRegression` class initializes weights and bias to zero. `fit(X, y)` computes predictions, mean squared error, and gradients for the weights and bias, then updates the parameters using the learning rate. `predict(X)` returns the linear predictions. The notebook defines `mse` and `r2_score` evaluation helpers.

## Run

Open the notebook in Jupyter and run cells from top to bottom. It generates 100 one-feature samples with `sklearn.datasets.make_regression` (noise 20, random state 4), then uses an 80/20 train/test split (random state 42).

## Output

The notebook plots training samples with the fitted line, plots the stored MSE values, prints test MSE and R², and prints the custom model's parameters beside scikit-learn's parameters.

## Implementation note

As currently written, `fit` first appends the initial cost repeatedly in a loop that does not update parameters, then appends cost again during the gradient-descent loop. Therefore `costs` has twice the configured number of iterations, and its first portion is a repeated initial value. The cost plot reflects this behavior.

## Dependencies

`numpy`, `matplotlib`, and `scikit-learn`.
