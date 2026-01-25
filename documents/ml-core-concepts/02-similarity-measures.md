# Similarity Measures in Machine Learning and AI

---
## 1. What is a Similarity Measure?

A similarity measure quantifies how alike two objects are.

* Higher value → more similar
* Lower value → less similar

**Intuition:**

* Distance asks “how far?
* ”Similarity asks “how related?”

---

## 2. Why Similarity Measures Matter

Similarity measures are used when:

* Ranking is more important than absolute distance
* Direction, overlap, or pattern matters more than geometry

Used in:

* NLP & embeddings
* Attention mechanisms
* Recommendation systems
* Search & retrieval
* Graphs & clustering

---

## 3. What is a similarity matrix?

If you have **N items**, similarity matrix (S) is an **N × N table**:

$
S_{ij} = \text{similarity(item i, item j)}
$

Example with 3 items A, B, C:

$
S =
\begin{bmatrix}
sim(A,A) & sim(A,B) & sim(A,C) \\
sim(B,A) & sim(B,B) & sim(B,C) \\
sim(C,A) & sim(C,B) & sim(C,C)
\end{bmatrix}
$

### Key properties (usually)

* Diagonal is max: (sim(A,A)) is highest (often 1)
* Often symmetric: (S_{ij} = S_{ji}) (not always)

---

## 4. Common Similarity Measures 

Here are the **most common similarity types** in ML/AI:

---

### 4.1. Cosine Similarity (most common for embeddings/NLP)

**Formula**:

$\text{cos}(x,y) = \frac{x \cdot y}{|x||y|}$

* $(x\cdot y)$ = dot product
* $(|x|)$ = vector length

**Intuition**:

* Measures angle between vectors
* Ignores magnitude, focuses on direction
* Two long vs short vectors pointing the same way → very similar

**Think**: “Do these vectors point in the same direction?”

**Pros**:

* Scale invariant
* Works extremely well with sparse data
* Very stable for high-dimensional embeddings

**Cons**:

* Ignores magnitude (can be bad if magnitude matters)
* Undefined for zero vectors (needs handling)

**When to use**:

* NLP embeddings (TF-IDF, Word2Vec, BERT)
* Sentence / document similarity
* Recommendation systems

**Scenario**:

Two documents:
* Doc x: (1, 2, 0)
* Doc y: (2, 4, 0)

**Calculation**: 

Dot product:

$
x\cdot y = 1\cdot2 + 2\cdot4 = 2 + 8 = 10
$

Norms:

$
|x|=\sqrt{1^2+2^2}=\sqrt{5},\quad |y|=\sqrt{2^2+4^2}=\sqrt{20}
$

Cosine similarity:

$
\frac{10}{\sqrt{5}\sqrt{20}}=\frac{10}{\sqrt{100}}=\frac{10}{10}=1
$

**Meaning**: Same topic, different length → cosine similarity = 1

---

### 4.2 Dot Product Similarity (used in Transformers Attention)

**Formula**:

$
sim(x,y) = x \cdot y
$

**Intuition**:

* Measures alignment + magnitude
* Bigger vectors aligned → larger score

**Think**: “How strongly do these vectors interact?”

**Pros**:

* Very fast to compute
* Works naturally with neural networks
* No normalization needed

**Cons**:

* Sensitive to scale
* Can explode with large vector values

**When to use**:

* Transformer attention
* Deep learning models
* Learned embeddings

**Scenario**:

**In attention**, they build a similarity matrix like:

$
\text{scores} = QK^T
$

Each entry = similarity between a **query token** and **key token**.

---

### 3.3. Jaccard Similarity (for sets/binary features)

**Formula**:

$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$

**Intuition**:

* Measures overlap vs total
* Ignores how many times something appears

**Think**: “How much do these sets overlap?”

**Pros**:

* Very intuitive
* Works well for binary/set data
* Ignores shared absences (important!)

**Cons**:

* Ignores frequency
* Not suitable for dense numeric data

**When to use**:

* Binary features
* Set comparisons
* Market-basket analysis

**Scenario**:

Shopping carts:

* A = {milk, bread, eggs}
* B = {bread, eggs, butter}

**Calculation**:

* Intersection = {bread, eggs} → 2
* Union = {milk, bread, eggs, butter} → 4

$
J=\frac{2}{4}=0.5
$

**Meaning**: 50% overlap

---

### 3.4. RBF / Gaussian Similarity (turns distance into similarity)

Very common in **kernels / spectral clustering / SVM kernels**.

**Formula**:

$
sim(x,y)=\exp\left(-\frac{|x-y|^2}{2\sigma^2}\right)
$

**Intuition**:

* Converts distance → similarity
* Nearby points → similarity close to 1
* Far points → similarity close to 0

**Think**: “Closer = exponentially more similar”

**Pros**:

* Smooth and well-behaved
* Works nicely with kernels
* Good for clustering & graphs

**Cons**:

* Requires tuning (\sigma)
* Computationally expensive for large datasets

**When to use**:

* Kernel methods (SVMs)
* Spectral clustering
* Graph-based learning

**Scenario**:

Two points:

* Distance = 0 → similarity = 1
* Distance = large → similarity ≈ 0

---

### 3.5. Pearson Correlation Similarity (patterns over time)

Used when you care about “shape/pattern” not scale.

### Formula

$
corr(x,y)=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2}\sqrt{\sum (y_i-\bar{y})^2}}
$

**Intuition**:

* Measures pattern similarity
* Ignores absolute values

**Think**: “Do these rise and fall together?”

**Pros**

* Captures trends
* Scale and shift invariant

**Cons**:

* Sensitive to noise
* Only captures linear relationships

**When to use**:

* Time-series data
* User rating patterns
* Recommender systems

**Scenario**:

Two users rate movies differently but like/dislike the same movies → high correlation

---

## 5. Distance vs Similarity — are they convertible?

Often yes.

### Common conversions

If you have distance (d(x,y)), you can convert to similarity like:

**Option 1 (simple):**
$
sim = \frac{1}{1 + d}
$

**Option 2 (Gaussian/RBF, best behaved):**
$
sim = \exp(-d^2 / 2\sigma^2)
$

So distance and similarity are related — but not identical.

---

## 6. A similarity matrix with a real calculation (step-by-step)

Let’s build a **cosine similarity matrix** for 3 items:

* $A=(1,0)$
* $B=(0,1)$
* $C=(1,1)$

Compute pairwise cosine similarity:

### $sim(A,B)$

$
A\cdot B = 0,;|A|=1,;|B|=1 \Rightarrow sim=0
$

### $sim(A,C)$

$
A\cdot C = 1, |A|=1, |C|=\sqrt{2} \Rightarrow sim=\frac{1}{\sqrt{2}} \approx 0.707
$

### $sim(B,C)$

Same result: $(\approx 0.707)$

So matrix:

$
S=
\begin{bmatrix}
1 & 0 & 0.707 \\
0 & 1 & 0.707 \\
0.707 & 0.707 & 1
\end{bmatrix}
$

That’s a similarity matrix.

---

## 7. Where it’s used (in simple terms)

* **KNN / search / retrieval**: find nearest or most similar items
* **Clustering (spectral)**: uses similarity matrix as graph edges
* **Recommenders**: item-item similarity or user-user similarity
* **Transformers (attention)**: similarity matrix between tokens (QKᵀ)

---
## 8. Similarity vs Distance (Quick View)
| Aspect        | Distance      | Similarity         |
| ------------- | ------------- | ------------------ |
| Meaning       | Dissimilarity | Affinity           |
| Smaller value | Closer        | Less similar       |
| Larger value  | Farther       | More similar       |
| Used in       | Geometry      | Ranking, attention |


---

## 9. Quick rule for choosing similarity type

* **Embeddings / text / images** → cosine or dot product
* **Set/binary features** → Jaccard
* **You already use Euclidean distance but need similarity** → RBF/Gaussian
* **Time series / trends** → correlation
