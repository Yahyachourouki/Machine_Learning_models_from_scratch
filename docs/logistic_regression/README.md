# Logistic Regression

This notebook implements binary logistic regression with a sigmoid function and gradient descent.

## File

- [`../../Logistic_Regression.ipynb`](../../Logistic_Regression.ipynb)

## Implementation

The `logistic_regression` class initializes weights and bias to zero. `fit(x, y)` calculates sigmoid probabilities, gradients, and parameter updates over `n_iters` iterations. `predict(x)` converts probabilities above 0.5 to class 1 and the rest to class 0.

## Run

Open the notebook in Jupyter and run cells from top to bottom. It loads scikit-learn's built-in breast cancer dataset and makes a 70/30 train/test split with `random_state=42`.

## Output

The notebook prints test accuracy and displays a confusion matrix labeled with the dataset's class names. The matrix compares actual labels with predicted labels; the diagonal cells are correct predictions and off-diagonal cells are errors.

## Implementation note

The notebook fits directly on the dataset's 30 input features without feature scaling. The code does not include a cost-history plot or probability output method.

## Dependencies

`numpy`, `matplotlib`, and `scikit-learn`.
