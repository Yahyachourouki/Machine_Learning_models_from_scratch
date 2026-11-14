# Logistic Regression Classifier

This notebook implements binary logistic regression using NumPy and gradient descent, then evaluates it on the Wisconsin breast cancer dataset.

## File and run

Open `Logistic_Regression.ipynb` in Jupyter and run cells in order. The notebook uses a 70/30 train/test split (`random_state=42`). Requirements: NumPy, Matplotlib, and scikit-learn (metrics and datasets).

## Mathematics

For an input \(x_i\), the model forms a linear score \(z_i=x_i^Tw+b\) and maps it to a probability with the sigmoid:

\[
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}=P(y_i=1\mid x_i).
\]

For binary labels \(y_i\in\{0,1\}\), the standard binary cross-entropy objective is

\[
J(w,b)=-\frac{1}{n}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
\]

Its gradients are \(\nabla_wJ=X^T(p-y)/n\) and \(\partial J/\partial b=\sum_i(p_i-y_i)/n\). The notebook applies gradient descent using these quantities. Prediction thresholds the sigmoid probability at 0.5: above 0.5 is class 1, otherwise class 0.

## Output and interpretation

The notebook prints test accuracy and displays a confusion matrix. Rows are actual labels and columns are predicted labels. Diagonal cells are correct classifications; off-diagonal cells show errors (false positives and false negatives). Dataset class names label the axes.

## Limitations

- The 30 input features are used without scaling. Feature magnitudes can affect optimization and the learning rate's suitability.
- The notebook does not retain training loss, return probabilities through a separate method, or apply regularization.
- The direct sigmoid formula may overflow for extreme scores; a numerically stable implementation is not used.
- Accuracy alone can hide which class is misclassified; inspect the confusion matrix as well.
