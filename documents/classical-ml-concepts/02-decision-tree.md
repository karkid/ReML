# Decision Tree

A **Decision Tree** is a supervised learning algorithm used for classification tasks.

It splits the dataset into smaller and smaller subsets based on **feature values**, forming a tree structure. Each **internal node** represents a feature test, each **branch** represents the outcome of that test, and each **leaf node** represents a final class label.

---

## Intuition

Decision Trees try to mimic human decision-making:
> "If temperature < 20°C → wear jacket; else if humidity > 80% → stay inside; else → go outside."

The tree repeatedly splits the data based on the feature that best separates the classes.

---

## How a Decision Tree Works

1. Start with the full dataset (root node).
2. For every feature and possible threshold, compute the **information gain** (how much uncertainty is reduced).
3. Choose the feature + threshold with the highest information gain.
4. Split the dataset accordingly into left/right subsets.
5. Recursively repeat for each subset until a stopping criterion is met:
   - Maximum depth reached
   - Node is pure (all one class)
   - Not enough samples left to split

---

## Key Concepts

### Entropy — Measuring Impurity

**Entropy** measures how mixed the class labels are in a dataset. If all samples belong to one class, entropy = 0 (pure). If classes are evenly split, entropy = 1 (maximum impurity).

$$
H(S) = - \sum_{i=1}^{k} p_i \log_2(p_i)
$$

| Class Distribution | Entropy |
|---|---|
| 100% one class | 0.0 |
| 50% / 50% | 1.0 |
| 70% / 30% | 0.88 |

### Threshold (Split Point)

A cutoff value for a continuous feature that divides the dataset into two subsets. The algorithm tests candidate thresholds per feature to find the one giving maximum information gain.

### Information Gain

Measures how much entropy decreases after a split:

$$
IG(S, A) = H(S) - \sum_{v \in \text{splits}} \frac{|S_v|}{|S|} H(S_v)
$$

**Goal:** maximize IG. The tree picks whichever (feature, threshold) pair does this best at each node.

### Worked Example

10 samples: 6 Yes, 4 No.

Parent entropy: $H(S) = -\frac{6}{10}\log_2(\frac{6}{10}) - \frac{4}{10}\log_2(\frac{4}{10}) = 0.97$

Split on "Temperature ≤ 20°C":

| Branch | Yes | No | Entropy |
|---|---|---|---|
| Left | 4 | 1 | 0.72 |
| Right | 2 | 3 | 0.97 |

Weighted child entropy: $\frac{5}{10}(0.72) + \frac{5}{10}(0.97) = 0.845$

Information Gain: $0.97 - 0.845 = 0.125$ bits.

---

## Stopping Criteria

`reml.tree.DecisionTree` stops growing a branch when any of the following holds:

- `depth > max_depth` (default `20`)
- `n_samples < min_samples_split` (default `2`)
- the node is pure (`n_labels == 1`)
- no split improves on the current node (falls back to a leaf)

Optionally, `n_features` caps how many features are considered per split — useful when this tree is used as a building block inside `RandomForest`, where each tree should see a random subset of features rather than all of them.

---

## Pros and Cons

| Advantages | Limitations |
|---|---|
| Easy to interpret & visualize | Prone to overfitting |
| Works with numerical & categorical data | Small changes in data can change splits |
| No feature scaling required | Greedy — locally optimal splits |
| Handles non-linear relationships | Can be biased with imbalanced data |

---

## Common Variants

| Algorithm | Description |
|---|---|
| ID3 | Uses information gain with entropy |
| C4.5 | Uses gain ratio (fixes bias of IG) |
| CART | Uses Gini impurity instead of entropy (used in scikit-learn) |
| Random Forest | Ensemble of many decision trees — see [`03-random-forest.md`](03-random-forest.md) |

`reml`'s implementation uses entropy + information gain, closest in spirit to ID3.

---

## Usage

```python
from reml.tree import DecisionTree

clf = DecisionTree(max_depth=10, min_samples_split=2)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
```

---

## Quick Summary

- **Entropy** measures impurity (uncertainty).
- **Information Gain** = how much entropy decreases after a split.
- **Threshold** determines where to split a feature.
- Implementation: [`reml/tree/DecisionTree.py`](../../reml/tree/DecisionTree.py)
