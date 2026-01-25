# Feature Scaling in Machine Learning

## What is a Scaler?

A **scaler** transforms features so that:

* values are on a comparable scale
* one feature doesn’t dominate others **just because of units**

**Intuition:**

> Scaling answers: *“How should numbers be represented so comparisons are fair?”*

This matters a LOT for:

* distance measures
* similarity measures
* gradient-based learning

---

## Why Scaling is Important

Imagine features:

| Feature | Range            |
| ------- | ---------------- |
| Age     | 0–100            |
| Salary  | 10,000–1,000,000 |

Without scaling:

* Salary dominates distance & gradients
* Age becomes almost irrelevant ❌

---

## 1. MinMaxScaler (Range Scaling)

**Formula**:

$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$

Scales values to **[0, 1]** (or any fixed range).

**Intuition**:

* Preserves **relative spacing**
* Just squeezes values into a box

**Think**: “Shrink everything proportionally into the same range.”

**Pros**:

* Keeps original distribution shape
* Bounded values (great for neural nets)
* Easy to interpret

**Cons**:

* **Very sensitive to outliers**
* New unseen values can break the range

**When to use**:

* Distance-based models (KNN, K-Means)
* Neural networks (especially sigmoid/tanh)
* Image pixel data

**Scenario**:

Feature: House price

* Min = 50k, Max = 500k
* Price = 275k

$
x' = \frac{275-50}{500-50} = 0.5
$

---

## 2. StandardScaler (Z-score Normalization)

**Formula**:

$
x' = \frac{x - \mu}{\sigma}
$

**Result**

* Mean = 0
* Std deviation = 1

**Intuition**:

* Measures **how many standard deviations away**
* Centers data around zero

**Think**: “How unusual is this value compared to average?”

**Pros**:

* Works well with Gaussian-like data
* Less sensitive to outliers than MinMax
* Ideal for gradient-based learning

**Cons**:

* Not bounded
* Still affected by extreme outliers

**When to use**:

* Linear regression
* Logistic regression
* SVM
* PCA
* Neural networks (ReLU-based)

**Scenario**:

Exam scores:

* Mean = 70, Std = 10
* Score = 85

$
x' = \frac{85 - 70}{10} = 1.5
$

Student is **1.5 std above average**

---

## 3. Normalizer (Vector Normalization)

### Formula (L2 norm)

$
x' = \frac{x}{|x|}
$

Each **row vector** gets unit length.

**Intuition**:

* Removes magnitude
* Keeps **direction only**

**Think**: “Only pattern matters, not size.”

**Pros**:

* Makes dot product = cosine similarity
* Essential for similarity & embeddings
* Scale-invariant

**Cons**:

* Destroys magnitude information
* Not suitable when absolute values matter

**When to use**:

* Cosine similarity
* Text embeddings
* NLP, recommender systems
* Attention with cosine similarity

**Scenario**:

Two documents:

* Doc A = (3,2)
* Doc B = (6,4)

After normalization:

* Both become same direction → similarity = 1

---

## 4. RobustScaler (Outlier-Resistant)

**Formula**:

$
x' = \frac{x - \text{median}}{IQR}
$

(IQR = Q3 − Q1)

**Intuition**:

* Uses **median instead of mean**
* Ignores extreme values

**Think**: “Scale based on typical data, not extremes.”

**Pros**:

* Robust to outliers
* Stable for noisy data

**Cons**:

* Less intuitive interpretation
* Not bounded

**When to use**:

* Financial data
* Sensor data
* Data with extreme outliers

**Scenario**:

Income data with billionaires → RobustScaler won’t get distorted

---

## 5. MaxAbsScaler

### Formula

$
x' = \frac{x}{|x|_{max}}
$

Scales to **[-1, 1]**

**Intuition**:

* Preserves sparsity
* No centering

**Pros**:

* Works well with sparse matrices
* Keeps zero entries intact

**Cons**:

* Sensitive to outliers
* No centering around zero

**When to use**:

* Sparse data
* Text data (TF-IDF)
* Large-scale linear models

---

# Scaling vs Normalization (Important!)

| Term          | What it does          |
| ------------- | --------------------- |
| Scaling       | Adjusts feature range |
| Normalization | Adjusts vector length |

They solve **different problems**.

---

# How Scaling Affects Distance & Similarity

### Without scaling

$
d(x,y) \approx \text{dominated by large-scale features}
$

### With scaling

$
d(x,y) = \text{fair comparison}
$

### Special case

* **Normalizer + dot product = cosine similarity**

---

# Quick Decision Guide

| Situation             | Use            |
| --------------------- | -------------- |
| KNN / K-Means         | MinMaxScaler   |
| Linear / Logistic Reg | StandardScaler |
| PCA                   | StandardScaler |
| NLP embeddings        | Normalizer     |
| Outliers present      | RobustScaler   |
| Sparse data           | MaxAbsScaler   |

---

## Final Takeaway

> **Scalers decide what “difference” and “similarity” mean numerically.**
> Bad scaling = wrong geometry = bad ML.
