# Logistic Regression

**Logistic Regression** is a classification algorithm used to predict the probability that an instance belongs to a particular class (binary: 0 or 1). Despite the name, it's used for classification, not regression.

---

## Intuition

We model the relationship between input features **X** and the probability of the positive class (**y = 1**) using the **sigmoid** function:

$$P(y=1|x) = \sigma(w^T x + b)$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- When `z` is large and positive → σ(z) ≈ 1
- When `z` is large and negative → σ(z) ≈ 0
- σ(z) maps any real number to the range (0, 1)

`reml`'s sigmoid is implemented piecewise — the positive-`z` and negative-`z` branches use algebraically-equivalent but differently-ordered forms of the same formula, specifically to avoid `exp()` overflow on large-magnitude inputs. A naive single-formula sigmoid will produce `inf`/`nan` on sufficiently extreme logits; this one won't.

---

## Decision Boundary

A prediction is made by comparing the probability to a threshold (0.5 by default, adjustable):

$$\hat{y} = \begin{cases} 1, & \sigma(w^Tx+b) \ge \text{threshold} \\ 0, & \text{otherwise} \end{cases}$$

---

## Loss Function

Trained by minimizing **Binary Cross-Entropy (log loss)**:

$$J(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Big]$$

`reml` clips predicted probabilities to `[1e-12, 1 - 1e-12]` before taking the log — `log(0)` is `-inf`, and a single mispredicted 0%-or-100%-confidence sample would otherwise blow up the whole loss.

---

## Gradient Descent Optimization

$$\frac{\partial J}{\partial w} = \frac{1}{N} X^T (\hat{y} - y), \qquad \frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

Updates: $w := w - \eta \cdot \frac{\partial J}{\partial w}$, $b := b - \eta \cdot \frac{\partial J}{\partial b}$

Same gradient-descent loop as [Linear Regression](04-linear-regression.md), with the sigmoid applied to the logits before computing error.

---

## Properties

| Property | Description |
|---|---|
| Type | Classification |
| Output | Probability (0–1) via `predict_proba`, or a hard label via `predict` |
| Loss function | Binary cross-entropy |
| Optimizer | Gradient descent |
| Decision surface | Linear (in feature space) |
| Complexity | O(N × D) per iteration |

---

## Evaluation

`reml.metrics` currently provides `accuracy_score` and `confusion_matrix` — both work with this model's `predict()` output. Precision, recall, and F1 are standard next steps for evaluating a classifier but aren't implemented in `reml.metrics` yet.

---

## Advantages

* Simple and efficient for binary classification.
* Outputs calibrated probabilities, not just labels.
* Easy to interpret coefficients.

## Limitations

* Not effective for non-linear decision boundaries.
* Sensitive to outliers and unscaled features.
* `reml`'s version has no built-in regularization (L1/L2) — a future addition, not currently present.

---

## Usage

```python
from reml.linear_model import LogisticRegression

model = LogisticRegression(learning_rate=0.001, n_iteration=1000)
model.fit(X_train, y_train)

labels = model.predict(X_test)             # hard 0/1 predictions
probs = model.predict_proba(X_test)        # [P(y=0), P(y=1)] per sample
labels_strict = model.predict(X_test, threshold=0.7)  # custom decision threshold
```

---

## Key Takeaways

- Logistic Regression = linear model + sigmoid + log loss.
- Convex optimization problem → gradient descent converges to a single global minimum.
- Feature scaling improves convergence speed and stability.
- Implementation: [`reml/linear_model/LogisticRegression.py`](../../reml/linear_model/LogisticRegression.py)
