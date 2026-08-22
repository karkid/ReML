# Linear Regression

Linear Regression is one of the simplest and most important algorithms in machine learning. It's used to **predict a continuous output** (like price, temperature, or score) based on one or more input features.

---

## Core Idea

> Linear Regression tries to find the best-fitting straight line (or plane) through the data.

The model assumes a linear relationship between input (X) and output (Y):

$$\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

or simply for one feature: $\hat{y} = wx + b$

---

## Weights, Bias, and Learning Rate

### Weights (w) — the slope, or each feature's influence

Each weight tells how much that feature contributes to the output. A large positive weight means increasing that feature increases the output a lot; a negative weight means the opposite.

Example: $\hat{y} = 5x + 2$ — here `w = 5` means "for every 1 unit increase in x, y increases by 5."

### Bias (b) — the offset / intercept

Shifts the prediction up or down; it's the model's prediction when all inputs are zero. In the example above, `b = 2` means the line crosses the Y-axis at 2.

### Learning Rate — how big each training step is

Controls how fast the model updates weights and bias. Too large, and the model jumps around and fails to converge. Too small, and learning is very slow — like adjusting your walking pace descending a hill: big steps risk overshooting the bottom, small steps take forever to get there.

---

## Visual Intuition

```
Y ↑
  |
  |          ●
  |      ●
  |   ●       ← line (model): ŷ = wx + b
  | ●
  +-------------------→ X
       slope = w
       intercept = b
```

---

## Objective

The goal is to find the values of `w` and `b` that make predictions as close as possible to the real outputs, measured by **Mean Squared Error (MSE)**:

$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

`reml`'s implementation tracks a **half-MSE** loss (`0.5 × mean squared error`) instead — this is a deliberate convention, not a bug: it makes the gradient expressions come out clean as `(1/n) · Xᵀ(ŷ − y)` with no stray factor of 2, which is why the formulas below match the code exactly.

---

## How the Model Learns — Gradient Descent

At each of `n_iteration` steps:

1. Predict: $\hat{y} = Xw + b$
2. Compute the error: $\hat{y} - y$
3. Compute gradients: $\frac{\partial J}{\partial w} = \frac{1}{n} X^T(\hat{y} - y)$, $\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)$
4. Update: $w := w - \text{lr} \cdot \frac{\partial J}{\partial w}$, $b := b - \text{lr} \cdot \frac{\partial J}{\partial b}$
5. Repeat.

**Numerical safety:** `reml`'s version clips gradients to `±1e6` before applying an update, and breaks out of the training loop early if a weight or bias update turns out `NaN`/`inf` — a diverging learning rate stops the fit instead of silently corrupting the model with garbage weights.

---

## Geometric Intuition

Imagine standing on a hill representing the loss surface. The goal is the lowest point (minimum loss):

* The slope of the hill = gradient.
* Your step size = learning rate.
* The direction you move = negative gradient.
* The position you adjust = weights and bias.

Over time, smaller effective steps bring you to the bottom, where loss is minimal.

---

## Usage

```python
from reml.linear_model import LinearRegression

model = LinearRegression(learning_rate=0.001, n_iteration=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(model.losses[-1])  # final training loss, tracked every iteration
```

`X` can be 1D (a single feature) or 2D (multiple features) — the model reshapes 1D input automatically.

---

## Summary Table

| Concept | Symbol | Meaning |
|---|---|---|
| Weight | `w` | Controls slope / feature importance |
| Bias | `b` | Shifts line up/down |
| Learning rate | `learning_rate` | Step size for learning |
| Prediction | `ŷ = Xw + b` | Estimated output |
| Loss function | half-MSE | Measures prediction error |
| Optimization | Gradient descent, with gradient clipping | Updates `w` and `b` to minimize loss |

Implementation: [`reml/linear_model/LinearRegression.py`](../../reml/linear_model/LinearRegression.py)
