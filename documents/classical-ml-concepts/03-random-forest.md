# Random Forest

A **Random Forest** is an ensemble of multiple decision trees combined to improve accuracy and reduce overfitting.

---

## Intuition

* A single decision tree can easily overfit training data.
* A Random Forest builds many trees, each on a random subset of the data.
* The final prediction is made by **majority vote** across all trees.

---

## How It Works

1. Draw random samples, with replacement, from the training data — **bootstrap sampling**.
2. Train a separate [`DecisionTree`](02-decision-tree.md) on each bootstrap sample.
3. At prediction time, run the input through every tree and take a majority vote across their outputs.

---

## Key Pieces

### `reml.utils.resample.bootstrap_sampling(X, y)`

Shared utility (used by `RandomForest.fit`, not reimplemented per-tree) that draws a bootstrap sample — random rows of `X`/`y`, sampled with replacement, same size as the original dataset.

### `RandomForest.fit(X, y)`

For each of `n_trees`, draws a fresh bootstrap sample and fits a new `DecisionTree` on it. Trees are independent — none of them see the same exact sample.

### `RandomForest.predict(X)`

Collects a prediction from every tree, then returns the most common class per sample (majority vote).

---

## Parameters

* **`n_trees`** (default `10`) — number of decision trees in the ensemble.
* **`max_depth`** (default `20`) — maximum depth of each individual tree.
* **`min_samples_split`** (default `2`) — minimum samples required to split a node.
* **`n_features`** (default `None`, i.e. all features) — features considered per split in each tree; passed straight through to each `DecisionTree`.

---

## Advantages

* Reduces overfitting compared to a single decision tree.
* Handles missing or unscaled features well (inherits this from `DecisionTree`).

## Limitations

* Slower to train and predict than one tree — training cost scales with `n_trees`.
* Harder to interpret than a single tree; you lose the readable if/else structure.

---

## Usage

```python
from reml.tree import RandomForest

clf = RandomForest(n_trees=50, max_depth=10)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
```

---

## Quick Summary

- Bootstrap sample → train a tree → repeat `n_trees` times → majority vote.
- Implementation: [`reml/tree/RandomForest.py`](../../reml/tree/RandomForest.py)
- Sampling utility: [`reml/utils/resample.py`](../../reml/utils/resample.py)
