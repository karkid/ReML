# k-Nearest Neighbors (kNN)

A simple and intuitive machine learning algorithm for **classification** based on the idea of *similarity*.

---

## What is kNN?

* **k-Nearest Neighbors (kNN)** is a **supervised learning** algorithm.
* It doesn't build an explicit model — instead, it **memorizes** the training data.
* To predict a new point, it looks at the **k nearest data points** in the training set and uses them to decide the output.

---

## Basic Idea

> "Tell me who your neighbors are, and I'll tell you who you are."

For a new data point:

1. Measure how close it is to all training points (using **distance**).
2. Pick the **k** closest points — the "nearest neighbors".
3. Take a **majority vote** of the neighbors' labels (or a **distance-weighted vote** — see below).

---

## Simple Diagram

```
y ↑
  |       🟥   🟥
  |   🟦   🟦      ?  ← new point
  |       🟥   🟦
  +------------------→ x

If k = 3 → two 🟥 and one 🟦 → predict 🟥
```

---

## Steps of kNN

1. **Choose k** → number of neighbors to look at.
2. **Compute distance** between the test point and every training point.
3. **Sort distances** and select the k nearest neighbors.
4. **Vote** (uniform or distance-weighted) on their labels.
5. **Return prediction**.

---

## Common Distance Metrics

| Metric | Formula (for points x₁ and x₂) | When to Use |
|---|---|---|
| **Euclidean** | √Σ(x₁ᵢ − x₂ᵢ)² | Most common for continuous data |
| **Manhattan** | Σ\|x₁ᵢ − x₂ᵢ\| | Good when outliers or grid-like data |
| **Minkowski** | (Σ\|x₁ᵢ − x₂ᵢ\|ᵖ)^(1/p) | General form (p=1 → Manhattan, p=2 → Euclidean) |

> Always **scale features** before using distance (e.g., normalize or standardize).

`reml`'s implementation lives in [`reml/spatial/distance.py`](../../reml/spatial/distance.py) as a shared module — every distance-based algorithm in the library calls the same `distance.euclidean()`, rather than each classifier reimplementing it.

---

## Choosing the Value of k

* **Small k (like 1 or 3):** more sensitive to noise, can overfit.
* **Large k (like 10 or 20):** smoother decision boundary, can underfit.
* A good starting point: `k ≈ √(number of training samples)`.
* Use odd k to avoid ties in binary classification.

---

## Uniform vs. Distance-Weighted Voting

`reml.neighbors.KNeighborsClassifier` supports two voting modes via `weights`:

- `weights="uniform"` (default `k=5`) — every one of the k neighbors gets an equal vote; the majority label wins.
- `weights="distance"` — closer neighbors count more. Each neighbor's vote is weighted by `1 / distance`, so a very close point can outweigh several farther ones. Useful when nearby points are genuinely more relevant than the rest of the neighborhood.

```python
from reml.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(k=5, weights="distance")
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)  # per-class probability estimates
```

`predict_proba` returns, for each sample, the fraction of the k neighbors (or weighted share, under `weights="distance"`) belonging to each class — useful when you need a confidence score rather than a hard label.

---

## Performance and Complexity

| Step | Complexity |
|---|---|
| **Training** | O(1) — just stores the data |
| **Prediction** | O(N × d) — N training samples, d features |
| **Memory** | Stores the full training set |

Can be slow for large datasets — every prediction checks all training points.

---

## Advantages

* Very easy to understand and implement.
* Works well for non-linear data.
* Naturally handles multi-class problems.
* No training phase needed.

---

## Limitations

* Slow at prediction for large datasets.
* Sensitive to feature scaling and irrelevant features.
* Curse of dimensionality — performance drops when features are too many.
* Doesn't handle missing data directly.

---

## Quick Summary

| Concept | Description |
|---|---|
| Algorithm type | Supervised, non-parametric, instance-based |
| Key hyperparameters | `k`, `weights` (`"uniform"` \| `"distance"`) |
| Distance metric | Euclidean (shared via `reml.spatial.distance`) |
| Training time | Very fast (just stores data) |
| Prediction time | Slower (distance computation per query) |
| Implementation | [`reml/neighbors/KNeighborsClassifier.py`](../../reml/neighbors/KNeighborsClassifier.py) |
