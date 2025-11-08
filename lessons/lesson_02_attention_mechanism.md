# Transformer Deep Dive: Stage 3A
## Single-Head Attention Mechanism

---

## Introduction: The Heart of the Transformer

After Stage 1 (embeddings) and Stage 2 (positional encoding), each token in our sequence has a rich, position-aware representation. But these representations are still **independent**—each token knows what it is and where it is, but nothing about the other tokens around it.

Stage 3 changes everything. The **Attention** mechanism allows each token to:
- Look at all other tokens in the sequence
- Decide which tokens are relevant
- Gather information from those relevant tokens
- Update its representation accordingly

This is where the transformer learns to understand **relationships** between tokens. For `Max(1,6,2)`, this is where the model learns that "Max" needs to look at "1", "6", and "2" to produce the correct answer.

---

## The Fundamental Problem: Contextual Understanding

Consider our sequence after Stage 2:
```
Position 0: "Max"  → [embedding with position info]
Position 1: "("    → [embedding with position info]
Position 2: "1"    → [embedding with position info]
Position 3: ","    → [embedding with position info]
Position 4: "6"    → [embedding with position info]
Position 5: ","    → [embedding with position info]
Position 6: "2"    → [embedding with position info]
Position 7: ")"    → [embedding with position info]
```

Each representation is isolated. The "Max" token doesn't "know" that it's followed by three numbers. The "6" doesn't "know" it's part of a Max operation.

**What we need:** A mechanism that lets each position gather context from all other positions.

**The solution:** Attention.

---

## The Core Concept of Attention

Attention computes a **weighted average** of all positions, where the weights are learned based on relevance.

**Intuitive Formula:**
```
For each position i:
  1. Compute relevance scores with all positions j
  2. Convert scores to probabilities (softmax)
  3. Take weighted average of all positions using these probabilities
```

**Mathematical Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
  Q = Query matrix    (what am I looking for?)
  K = Key matrix      (what information do I have?)
  V = Value matrix    (what information do I give?)
```

This looks cryptic, so let's build it up step by step with concrete numbers.

---

## Step 1: Creating Query, Key, and Value

The first step is to transform our input embeddings into three different "views" of the same information.

### Input Dimensions

**Input:**
- X ∈ ℝ^(8 × 64) — our sequence after positional encoding
- 8 positions, each with 64-dimensional embeddings

**Example input for position 0 ("Max"):**
```
X[0] = [-0.87, -0.59, -0.29, 0.12, 0.45, -0.23, ..., 1.09]  (64 values)
```

**Example input for position 4 ("6"):**
```
X[4] = [-0.87, -0.59, -0.29, 0.34, 0.67, -0.45, ..., 1.09]  (64 values)
```

### Linear Projections

We multiply by three different learned weight matrices:

```
Q = X W_q    where W_q ∈ ℝ^(64 × d_k)
K = X W_k    where W_k ∈ ℝ^(64 × d_k)
V = X W_v    where W_v ∈ ℝ^(64 × d_v)
```

For a single attention head with d_k = d_v = 16:
```
Q = X W_q    →  (8 × 64) @ (64 × 16) = (8 × 16)
K = X W_k    →  (8 × 64) @ (64 × 16) = (8 × 16)
V = X W_v    →  (8 × 64) @ (64 × 16) = (8 × 16)
```

**Result:** Each position now has three 16-dimensional vectors.

### Concrete Example

**For position 0 ("Max"):**
```
X[0] = [-0.87, -0.59, -0.29, ..., 1.09]  (64 dims)

Step 1: Multiply by W_q
Q[0] = X[0] @ W_q 
     = [-0.87×W_q[0,0] + -0.59×W_q[1,0] + ... + 1.09×W_q[63,0],
        -0.87×W_q[0,1] + -0.59×W_q[1,1] + ... + 1.09×W_q[63,1],
        ...
        -0.87×W_q[0,15] + -0.59×W_q[1,15] + ... + 1.09×W_q[63,15]]
     = [0.23, -0.45, 0.12, 0.34, -0.67, 0.89, ..., 0.67]  (16 dims)

Step 2: Multiply by W_k
K[0] = X[0] @ W_k 
     = [0.34, -0.12, 0.56, 0.23, -0.45, 0.78, ..., -0.23] (16 dims)

Step 3: Multiply by W_v
V[0] = X[0] @ W_v 
     = [0.45, -0.23, 0.78, 0.12, -0.56, 0.34, ..., 0.34]  (16 dims)
```

**For position 4 ("6"):**
```
X[4] = [-0.87, -0.59, -0.29, ..., 1.09]  (64 dims)

Q[4] = X[4] @ W_q = [0.56, -0.23, 0.34, 0.45, -0.12, 0.67, ..., 0.12]  (16 dims)
K[4] = X[4] @ W_k = [0.23, -0.56, 0.45, 0.12, -0.34, 0.89, ..., -0.34] (16 dims)
V[4] = X[4] @ W_v = [0.67, -0.34, 0.23, 0.45, -0.78, 0.12, ..., 0.56]  (16 dims)
```

### Why Three Different Matrices?

The separation into Q, K, V gives the model flexibility:

**Query (Q):** Represents what this position is "asking for"
- Position 0 ("Max") might query: "Give me information about numerical values"

**Key (K):** Represents what this position "offers"
- Position 4 ("6") might offer: "I am a numerical value: 6"

**Value (V):** Represents the actual information to be passed
- Position 4's value might encode: "I'm the largest number in this sequence"

### The Database Analogy

Think of it like a database query system:

```
Query:    "Find all products with price > $100"
Keys:     Product1: "electronics, $150"
          Product2: "books, $20"
          Product3: "electronics, $200"
Values:   Product1: [detailed info about Product1]
          Product2: [detailed info about Product2]
          Product3: [detailed info about Product3]

Result: Match Query against Keys → Find Product1, Product3
        Return weighted combination of their Values
```

In attention:
```
Position 0 (Max): Query = "What numbers should I compare?"
Position 2 (1):   Key = "I'm a number: 1",    Value = [representation of 1]
Position 4 (6):   Key = "I'm a number: 6",    Value = [representation of 6]
Position 6 (2):   Key = "I'm a number: 2",    Value = [representation of 2]

Result: Max's query matches strongly with positions 2, 4, 6
        Return weighted average of their values
```

### Why Not Use Input Directly?

**Bad approach:** Use X directly without Q, K, V projections
```
Attention = X @ X^T  (just dot products of inputs)
```

**Problems:**
1. **No flexibility:** Can't separately represent "what I'm looking for" vs "what I offer"
2. **No learning:** No parameters to adjust based on the task
3. **Wrong dimensionality:** 64×64 attention matrix, computationally expensive
4. **No specialization:** Can't learn task-specific relevance patterns

**Good approach:** Use learned projections Q, K, V
```
Q = X W_q  ← learns what to ask for
K = X W_k  ← learns what to advertise
V = X W_v  ← learns what to give
```

**Benefits:**
1. **Flexibility:** Separate roles for querying and offering information
2. **Learnable:** 3 × (64 × 16) = 3,072 parameters to optimize
3. **Efficient:** 16-dimensional subspace instead of full 64
4. **Task-specific:** Learns patterns specific to Max/Min/First operations

---

## Step 2: Computing Attention Scores

Now we compute how relevant each position is to every other position using dot products.

### The Operation

```
Scores = Q K^T

Where:
  Q ∈ ℝ^(8 × 16)      (each row is a query)
  K^T ∈ ℝ^(16 × 8)    (each column is a key)
  Scores ∈ ℝ^(8 × 8)  (all pairwise scores)
```

### What Does This Compute?

Each element scores[i, j] is the **dot product** of Q[i] and K[j]:

```
scores[i, j] = Q[i] · K[j] 
             = Σ(Q[i, d] × K[j, d]) for d = 0 to 15
             = Q[i,0]×K[j,0] + Q[i,1]×K[j,1] + ... + Q[i,15]×K[j,15]
```

**Geometric interpretation:** Dot product measures similarity/alignment
- Large positive: Q[i] and K[j] point in same direction → high relevance
- Near zero: Q[i] and K[j] are orthogonal → low relevance
- Large negative: Q[i] and K[j] point in opposite directions → negative relevance

### Detailed Calculation Example

**Position 0 ("Max") attending to position 4 ("6"):**

```
Q[0] = [0.23, -0.45, 0.12, 0.34, -0.67, 0.89, 0.45, -0.23, 
        0.56, -0.12, 0.34, 0.78, -0.45, 0.23, 0.67, -0.34]

K[4] = [0.23, -0.56, 0.45, 0.12, -0.34, 0.89, 0.34, -0.45,
        0.67, -0.23, 0.12, 0.56, -0.67, 0.34, 0.45, -0.23]

scores[0, 4] = Q[0] · K[4]
             = 0.23×0.23 + (-0.45)×(-0.56) + 0.12×0.45 + 0.34×0.12
             + (-0.67)×(-0.34) + 0.89×0.89 + 0.45×0.34 + (-0.23)×(-0.45)
             + 0.56×0.67 + (-0.12)×(-0.23) + 0.34×0.12 + 0.78×0.56
             + (-0.45)×(-0.67) + 0.23×0.34 + 0.67×0.45 + (-0.34)×(-0.23)
             
             = 0.053 + 0.252 + 0.054 + 0.041
             + 0.228 + 0.792 + 0.153 + 0.104
             + 0.375 + 0.028 + 0.041 + 0.437
             + 0.302 + 0.078 + 0.302 + 0.078
             
             = 3.318
```

**Position 4 ("6") attending to position 0 ("Max"):**

```
Q[4] = [0.56, -0.23, 0.34, 0.45, -0.12, 0.67, 0.23, -0.45,
        0.78, -0.34, 0.45, 0.23, -0.56, 0.12, 0.34, -0.67]

K[0] = [0.34, -0.12, 0.56, 0.23, -0.45, 0.78, 0.12, -0.34,
        0.45, -0.23, 0.34, 0.12, -0.78, 0.23, 0.45, -0.12]

scores[4, 0] = 0.56×0.34 + (-0.23)×(-0.12) + ... + (-0.67)×(-0.12)
             = 2.987
```

**Note:** scores[0, 4] ≠ scores[4, 0] in general! The attention matrix is **not symmetric**.

### Full Scores Matrix

Computing all 64 pairwise dot products:

```
           0     1     2     3     4     5     6     7
        (Max)  (  )  (1)  (,)  (6)  (,)  (2)  ()
    0 [  4.2   1.8   0.9   0.4   3.3   0.5   1.2   2.1 ]  Max
    1 [  2.7   3.8   1.4   0.7   2.1   0.9   1.5   3.0 ]  (
    2 [  1.5   2.3   4.5   1.0   2.5   1.2   1.9   1.4 ]  1
    3 [  0.6   1.0   1.6   3.2   0.7   2.1   0.9   1.2 ]  ,
    4 [  3.0   1.5   1.9   0.9   4.8   1.4   2.3   1.0 ]  6
    5 [  0.7   1.2   1.8   2.7   1.0   3.8   1.4   1.6 ]  ,
    6 [  2.1   1.4   2.3   0.7   2.8   1.2   4.3   0.9 ]  2
    7 [  1.8   2.5   1.2   1.6   1.9   1.0   1.4   3.9 ]  )
```

### Interpretation of Scores

**Row i = Position i is querying** (what this position wants)
**Column j = Position j is being attended to** (how relevant j is to i)

**Position 0 (Max) scores:**
```
[4.2, 1.8, 0.9, 0.4, 3.3, 0.5, 1.2, 2.1]
```
- Highest with itself (4.2): Self-attention is strong
- High with "6" at position 4 (3.3): Max operation needs number values
- Moderate with ")" at position 7 (2.1): End of expression marker
- Low with commas (0.4, 0.5): Syntax tokens less relevant

**Position 4 ("6") scores:**
```
[3.0, 1.5, 1.9, 0.9, 4.8, 1.4, 2.3, 1.0]
```
- Highest with itself (4.8): Preserving own information
- High with "Max" at position 0 (3.0): Needs to know operation type
- Moderate with "2" at position 6 (2.3): Other number for comparison
- Moderate with "1" at position 2 (1.9): Other number for comparison
- Lower with syntax tokens

### Why Are Diagonal Values High?

Every position has the highest score with itself because Q[i] and K[i] are derived from the same input X[i]:

```
Q[i] = X[i] @ W_q
K[i] = X[i] @ W_k

Q[i] · K[i] = (X[i] @ W_q) · (X[i] @ W_k)
            = X[i]^T W_q^T W_k X[i]
```

This is typically large because:
1. X[i] appears in both terms
2. W_q and W_k are randomly initialized, but typically result in positive correlations
3. This is actually **useful**: each position should consider its own information!

Self-attention helps preserve the original information while adding context from other positions.

---

## Step 3: Scaling by √d_k

### The Problem with Large Scores

Dot products can grow very large as dimensionality increases:

**For d_k = 16:**
```
If Q[i] and K[j] have values around ±1:
  Each term: Q[i,d] × K[j,d] ≈ -1 to +1
  Sum of 16 terms: scores[i, j] ≈ -16 to +16 (approximately)
```

**Example extreme case:**
```
Q[i] = [1.0, 1.0, 1.0, ..., 1.0]  (all 1's, 16 dims)
K[j] = [1.0, 1.0, 1.0, ..., 1.0]  (all 1's, 16 dims)

scores[i, j] = 1×1 + 1×1 + ... + 1×1 (16 times)
             = 16.0
```

**Why is this a problem?**

Large scores cause problems in the next step (softmax). Consider:

```
scores = [0.5, 1.0, 15.0]

exp(0.5) = 1.65
exp(1.0) = 2.72
exp(15.0) = 3,269,017  ← Explodes!

softmax = [0.0000005, 0.0000008, 0.9999987]
          ↑                       ↑
    Effectively zero         Effectively one
```

When one score is much larger, softmax becomes almost "one-hot":
- All weight goes to one position
- Gradients vanish for other positions (derivative ≈ 0)
- Model can't learn to attend to multiple relevant positions

### The Solution: Scale by √d_k

Divide all scores by √d_k:

```
scaled_scores = scores / √d_k

For d_k = 16:
  √d_k = √16 = 4

scaled_scores = scores / 4
```

### After Scaling

```
Original scores:
           0     1     2     3     4     5     6     7
    0 [  4.2   1.8   0.9   0.4   3.3   0.5   1.2   2.1 ]
    1 [  2.7   3.8   1.4   0.7   2.1   0.9   1.5   3.0 ]
    2 [  1.5   2.3   4.5   1.0   2.5   1.2   1.9   1.4 ]
    3 [  0.6   1.0   1.6   3.2   0.7   2.1   0.9   1.2 ]
    4 [  3.0   1.5   1.9   0.9   4.8   1.4   2.3   1.0 ]
    5 [  0.7   1.2   1.8   2.7   1.0   3.8   1.4   1.6 ]
    6 [  2.1   1.4   2.3   0.7   2.8   1.2   4.3   0.9 ]
    7 [  1.8   2.5   1.2   1.6   1.9   1.0   1.4   3.9 ]

Scaled scores (÷ 4):
           0     1     2     3     4     5     6     7
    0 [  1.05  0.45  0.23  0.10  0.83  0.13  0.30  0.53 ]
    1 [  0.68  0.95  0.35  0.18  0.53  0.23  0.38  0.75 ]
    2 [  0.38  0.58  1.13  0.25  0.63  0.30  0.48  0.35 ]
    3 [  0.15  0.25  0.40  0.80  0.18  0.53  0.23  0.30 ]
    4 [  0.75  0.38  0.48  0.23  1.20  0.35  0.58  0.25 ]
    5 [  0.18  0.30  0.45  0.68  0.25  0.95  0.35  0.40 ]
    6 [  0.53  0.35  0.58  0.18  0.70  0.30  1.08  0.23 ]
    7 [  0.45  0.63  0.30  0.40  0.48  0.25  0.35  0.98 ]
```

Now scores are in a more reasonable range for softmax!

### Why Specifically √d_k?

This comes from **variance analysis**. Assume Q and K have:
- Mean = 0
- Variance = 1

For a single term:
```
Var(Q[i,d] × K[j,d]) = Var(Q[i,d]) × Var(K[j,d]) 
                      = 1 × 1 = 1
```

For the sum (dot product):
```
Var(Q[i] · K[j]) = Var(Σ Q[i,d] × K[j,d])
                 = Σ Var(Q[i,d] × K[j,d])  (assuming independence)
                 = d_k × 1
                 = d_k
```

So the dot product has variance d_k. To normalize back to unit variance:

```
Var((Q[i] · K[j]) / √d_k) = Var(Q[i] · K[j]) / d_k
                           = d_k / d_k
                           = 1 ✓
```

This keeps the distribution stable regardless of d_k!

**Comparison of different scalings:**

```
No scaling (/ 1):
  Range: [-16, 16]
  Std: 4.0
  Softmax: Often too sharp

Scale by d_k (/ 16):
  Range: [-1, 1]
  Std: 0.25
  Softmax: Often too uniform

Scale by √d_k (/ 4):
  Range: [-4, 4]
  Std: 1.0
  Softmax: Just right! ✓
```

---

## Step 4: Softmax (Converting Scores to Probabilities)

Now we convert scaled scores into proper probability distributions using the softmax function.

### The Softmax Formula

For each row i (each query position):

```
attention_weights[i, j] = exp(scaled_scores[i, j]) / Σ_k exp(scaled_scores[i, k])
```

This ensures:
1. All weights are positive: attention_weights[i, j] ∈ [0, 1]
2. All weights sum to 1: Σ_j attention_weights[i, j] = 1
3. Higher scores get exponentially more weight

### Step-by-Step Calculation

**For Position 4 (the "6"):**

```
Step 1: Extract the row of scaled scores
  scaled_scores[4] = [0.75, 0.38, 0.48, 0.23, 1.20, 0.35, 0.58, 0.25]

Step 2: Exponentiate each value
  exp(0.75) = 2.117
  exp(0.38) = 1.462
  exp(0.48) = 1.616
  exp(0.23) = 1.259
  exp(1.20) = 3.320  ← highest score
  exp(0.35) = 1.419
  exp(0.58) = 1.786
  exp(0.25) = 1.284

  exp_scores = [2.117, 1.462, 1.616, 1.259, 3.320, 1.419, 1.786, 1.284]

Step 3: Sum all exponentiated values
  sum = 2.117 + 1.462 + 1.616 + 1.259 + 3.320 + 1.419 + 1.786 + 1.284
      = 14.263

Step 4: Divide each by the sum
  attention[4, 0] = 2.117 / 14.263 = 0.148
  attention[4, 1] = 1.462 / 14.263 = 0.103
  attention[4, 2] = 1.616 / 14.263 = 0.113
  attention[4, 3] = 1.259 / 14.263 = 0.088
  attention[4, 4] = 3.320 / 14.263 = 0.233  ← highest weight
  attention[4, 5] = 1.419 / 14.263 = 0.099
  attention[4, 6] = 1.786 / 14.263 = 0.125
  attention[4, 7] = 1.284 / 14.263 = 0.090

  attention[4] = [0.148, 0.103, 0.113, 0.088, 0.233, 0.099, 0.125, 0.090]
```

**Verification:**
```
Sum = 0.148 + 0.103 + 0.113 + 0.088 + 0.233 + 0.099 + 0.125 + 0.090
    = 0.999 ≈ 1.000 ✓
```

(Small rounding errors are normal)

### Full Attention Weights Matrix

Applying softmax to all rows:

```
           0     1     2     3     4     5     6     7    Sum
        (Max)  (  )  (1)  (,)  (6)  (,)  (2)  ()
    0 [ 0.178 0.097 0.078 0.069 0.143 0.071 0.083 0.103 ] 1.00  Max
    1 [ 0.125 0.162 0.090 0.075 0.108 0.079 0.092 0.131 ] 1.00  (
    2 [ 0.090 0.113 0.193 0.081 0.119 0.085 0.101 0.089 ] 1.00  1
    3 [ 0.072 0.080 0.093 0.139 0.074 0.107 0.079 0.084 ] 1.00  ,
    4 [ 0.148 0.103 0.113 0.088 0.233 0.099 0.125 0.090 ] 1.00  6
    5 [ 0.074 0.084 0.096 0.123 0.079 0.162 0.090 0.099 ] 1.00  ,
    6 [ 0.108 0.090 0.113 0.074 0.128 0.084 0.184 0.079 ] 1.00  2
    7 [ 0.097 0.119 0.084 0.093 0.100 0.079 0.089 0.166 ] 1.00  )
```

### Interpretation

**Position 4 ("6") attention weights:**
```
[0.148, 0.103, 0.113, 0.088, 0.233, 0.099, 0.125, 0.090]
```

- **Most attention (23.3%):** Position 4 itself (the "6")
  - Preserves own information
  
- **Significant attention (14.8%):** Position 0 ("Max")
  - Needs to know what operation to perform
  
- **Moderate attention (12.5%):** Position 6 ("2")
  - Other number to compare against
  
- **Moderate attention (11.3%):** Position 2 ("1")
  - Another number to compare against
  
- **Low attention (~10% each):** Syntax tokens (parentheses, commas)
  - Less relevant for numerical comparison

This makes intuitive sense! The "6" pays most attention to:
1. Itself (preserve value)
2. The operation type (know what to do)
3. Other numbers (for comparison)
4. Less to syntax (structural markers)

### Properties of Softmax

**1. Probability distribution:**
```
All values in [0, 1]
Sum equals 1
Can interpret as "how much to attend to each position"
```

**2. Exponential amplification:**
```
Input scores:  [1.0,  1.5,  2.0]
After softmax: [0.186, 0.307, 0.507]

The largest input (2.0) gets disproportionately more weight (50.7%)
Not a linear transformation!
```

**3. Differentiable:**
```
Softmax has well-defined gradients for backpropagation:

∂softmax[i] / ∂x[j] = softmax[i] × (δ[i,j] - softmax[j])

Where δ[i,j] = 1 if i==j, else 0
```

**4. Temperature control (advanced):**
```
softmax(x / T) where T = temperature

T < 1: Sharpens distribution (more focused)
  softmax([1, 2, 3] / 0.5) = [0.09, 0.24, 0.67]  ← more extreme

T = 1: Standard softmax
  softmax([1, 2, 3] / 1.0) = [0.09, 0.24, 0.67]

T > 1: Flattens distribution (more uniform)
  softmax([1, 2, 3] / 2.0) = [0.19, 0.29, 0.52]  ← more balanced
```

**5. Handles negative scores:**
```
scores = [-2.0, 0.0, 2.0]

exp(-2.0) = 0.135
exp(0.0) = 1.000
exp(2.0) = 7.389

softmax = [0.016, 0.117, 0.867]

Negative scores just get low (but still positive) weights.
```

---

## Step 5: Weighted Sum of Values

Finally, we use the attention weights to compute a weighted average of the Value vectors. This is where information actually flows between positions.

### The Operation

```
Output = attention_weights @ V

Where:
  attention_weights ∈ ℝ^(8 × 8)
  V ∈ ℝ^(8 × 16)
  Output ∈ ℝ^(8 × 16)
```

### For Each Position

```
Output[i] = Σ_j (attention_weights[i, j] × V[j])

This is a weighted average of all Value vectors,
where weights come from attention.
```

### Detailed Calculation for Position 4

**Position 4 ("6") gathering context:**

```
Attention weights[4] = [0.148, 0.103, 0.113, 0.088, 0.233, 0.099, 0.125, 0.090]

Value vectors:
V[0] (Max) = [0.45, -0.23, 0.78, 0.12, -0.56, 0.34, 0.23, -0.45, 
              0.67, -0.12, 0.34, 0.56, -0.34, 0.23, 0.45, 0.34]

V[1] (() = [0.23, -0.56, 0.45, 0.34, -0.23, 0.67, 0.12, -0.34,
            0.45, -0.23, 0.12, 0.34, -0.56, 0.45, 0.23, -0.12]

V[2] (1) = [0.67, -0.34, 0.23, 0.45, -0.78, 0.12, 0.34, -0.23,
            0.56, -0.45, 0.23, 0.12, -0.45, 0.34, 0.67, 0.56]

V[3] (,) = [0.12, -0.45, 0.67, 0.23, -0.34, 0.56, 0.45, -0.12,
            0.34, -0.67, 0.45, 0.23, -0.12, 0.56, 0.34, -0.23]

V[4] (6) = [0.67, -0.34, 0.23, 0.45, -0.78, 0.12, 0.34, -0.23,
            0.56, -0.45, 0.23, 0.12, -0.45, 0.34, 0.67, 0.56]

V[5] (,) = [0.12, -0.45, 0.67, 0.23, -0.34, 0.56, 0.45, -0.12,
            0.34, -0.67, 0.45, 0.23, -0.12, 0.56, 0.34, -0.23]

V[6] (2) = [0.34, -0.12, 0.56, 0.23, -0.45, 0.78, 0.12, -0.34,
            0.45, -0.23, 0.34, 0.12, -0.78, 0.45, 0.23, -0.45]

V[7] ()) = [0.23, -0.34, 0.45, 0.12, -0.23, 0.67, 0.34, -0.45,
            0.12, -0.56, 0.23, 0.34, -0.45, 0.12, 0.56, 0.12]
```

**Computing Output[4]:**

For each dimension d = 0 to 15, we compute:
```
Output[4, d] = Σ_j (attention[4, j] × V[j, d])
```

**Dimension 0:**
```
Output[4, 0] = 0.148×0.45 + 0.103×0.23 + 0.113×0.67 + 0.088×0.12
             + 0.233×0.67 + 0.099×0.12 + 0.125×0.34 + 0.090×0.23
             
             = 0.067 + 0.024 + 0.076 + 0.011
             + 0.156 + 0.012 + 0.043 + 0.021
             
             = 0.410
```

**Dimension 1:**
```
Output[4, 1] = 0.148×(-0.23) + 0.103×(-0.56) + 0.113×(-0.34) + 0.088×(-0.45)
             + 0.233×(-0.34) + 0.099×(-0.45) + 0.125×(-0.12) + 0.090×(-0.34)
             
             = -0.034 + (-0.058) + (-0.038) + (-0.040)
             + (-0.079) + (-0.045) + (-0.015) + (-0.031)
             
             = -0.340
```

**Dimension 2:**
```
Output[4, 2] = 0.148×0.78 + 0.103×0.45 + 0.113×0.23 + 0.088×0.67
             + 0.233×0.23 + 0.099×0.67 + 0.125×0.56 + 0.090×0.45
             
             = 0.115 + 0.046 + 0.026 + 0.059
             + 0.054 + 0.066 + 0.070 + 0.041
             
             = 0.477
```

**... continuing for all 16 dimensions ...**

```
Output[4] = [0.410, -0.340, 0.477, 0.267, -0.512, 0.389, 0.298, -0.289,
             0.445, -0.398, 0.301, 0.256, -0.423, 0.367, 0.489, 0.234]
```

### What Just Happened?

Position 4's new representation is now a **mixture** of information from all positions:

```
23.3% from V[4] (itself - the "6")
14.8% from V[0] ("Max" operation)
12.5% from V[6] (the "2")
11.3% from V[2] (the "1")
~38% from other positions (syntax, etc.)
```

**Before attention:** Position 4 knew "I am the digit 6"

**After attention:** Position 4 knows:
- "I am the digit 6"
- "I'm part of a Max operation"
- "I'm being compared with 1 and 2"
- "I appear to be the largest value"

This is **contextual understanding** emerging from the attention mechanism!

### Complete Output for All Positions

After computing the weighted sum for all 8 positions:

```
Output[0] = [0.521, -0.267, 0.398, ...]  (16 dims) ← Max, now context-aware
Output[1] = [0.345, -0.412, 0.287, ...]  (16 dims) ← (, now context-aware
Output[2] = [0.423, -0.298, 0.456, ...]  (16 dims) ← 1, now context-aware
Output[3] = [0.289, -0.378, 0.312, ...]  (16 dims) ← ,, now context-aware
Output[4] = [0.410, -0.340, 0.477, ...]  (16 dims) ← 6, now context-aware
Output[5] = [0.301, -0.389, 0.334, ...]  (16 dims) ← ,, now context-aware
Output[6] = [0.378, -0.321, 0.445, ...]  (16 dims) ← 2, now context-aware
Output[7] = [0.356, -0.356, 0.401, ...]  (16 dims) ← ), now context-aware
```

Each position has gathered relevant context from other positions!

---

## Complete Flow: Tracing One Position

Let's trace the complete attention flow for position 4 ("6"):

```
STAGE 2 OUTPUT:
X[4] = [-0.87, -0.59, -0.29, ..., 1.09]  (64 dims)
       ↓

STEP 1: Linear Projections
Q[4] = X[4] @ W_q = [0.56, -0.23, 0.34, ..., 0.12]  (16 dims)
K[4] = X[4] @ W_k = [0.23, -0.56, 0.45, ..., -0.34] (16 dims)
V[4] = X[4] @ W_v = [0.67, -0.34, 0.23, ..., 0.56]  (16 dims)
       ↓

STEP 2: Compute Attention Scores (with all positions)
scores[4, :] = Q[4] · [K[0], K[1], ..., K[7]]
             = [3.0, 1.5, 1.9, 0.9, 4.8, 1.4, 2.3, 1.0]
       ↓

STEP 3: Scale by √16 = 4
scaled[4, :] = [0.75, 0.38, 0.48, 0.23, 1.20, 0.35, 0.58, 0.25]
       ↓

STEP 4: Softmax
attention[4, :] = softmax(scaled[4, :])
                = [0.148, 0.103, 0.113, 0.088, 0.233, 0.099, 0.125, 0.090]
       ↓

STEP 5: Weighted Sum of Values
Output[4] = 0.148×V[0] + 0.103×V[1] + ... + 0.090×V[7]
          = [0.410, -0.340, 0.477, ..., 0.234]  (16 dims)
```

**Result:** Position 4 started with a 64-dim isolated representation and now has a 16-dim context-aware representation that incorporates information from all relevant positions.

---

## Dimensionality Summary

Let's track how dimensions change through attention:

```
Input:  X ∈ ℝ^(8 × 64)
        8 positions, each 64 dimensions

        ↓ W_q, W_k, W_v (64 × 16 each)

Q, K, V: ℝ^(8 × 16)
         8 positions, each projected to 16 dimensions

        ↓ Q @ K^T

Scores: ℝ^(8 × 8)
        All pairwise relevance scores

        ↓ Scale, Softmax

Attention: ℝ^(8 × 8)
           Probability distributions (each row sums to 1)

        ↓ @ V

Output: ℝ^(8 × 16)
        8 positions, each 16 dimensions (context-aware)
```

**Key observation:** We went from 64 → 16 dimensions. This is efficient! Later (Stage 3B), we'll see how multi-head attention uses multiple 16-dim projections in parallel to get back to 64 total dimensions.

---

## Parameter Count

How many learnable parameters in single-head attention?

```
W_q: 64 × 16 = 1,024 parameters
W_k: 64 × 16 = 1,024 parameters
W_v: 64 × 16 = 1,024 parameters
────────────────────────────────
Total:         3,072 parameters
```

**No biases:** Standard attention doesn't use bias terms in the projections.

**No parameters in softmax:** Softmax is a fixed operation, no learned weights.

**Comparison to other layers:**
- Token embeddings (20 tokens × 64 dims): 1,280 parameters
- Single-head attention: 3,072 parameters
- Feed-forward network (later): ~16,000 parameters

Attention is parameter-efficient relative to its expressive power!

---

## Value Ranges Throughout Attention

Understanding magnitudes helps debug and interpret models:

**Input (X):**
- Range: approximately [-2, 2]
- Typically normalized or standardized

**Q, K, V after projection:**
- Range: approximately [-3, 3]
- Mean: close to 0
- Std: approximately 1

**Attention scores (Q @ K^T):**
- Range: approximately [-10, 10] for d_k = 16
- Unbounded, depends on alignment of Q and K

**Scaled scores (÷ √d_k):**
- Range: approximately [-2.5, 2.5]
- Designed for softmax stability
- Std: approximately 1

**Attention weights (after softmax):**
- Range: exactly [0, 1]
- Sum: exactly 1.0 per row
- Typically: most values between 0.05 and 0.30 for 8 positions

**Output values:**
- Range: approximately [-3, 3]
- Similar to input, but context-aware
- Mean: close to 0

---

## Gradient Flow (How Learning Happens)

During backpropagation, gradients flow backward through all operations:

### Backward Through Weighted Sum

```
Given: ∂Loss/∂Output

Compute:
  ∂Loss/∂attention_weights = ∂Loss/∂Output @ V^T
  ∂Loss/∂V = attention_weights^T @ ∂Loss/∂Output
```

**Each position's value gradient depends on:**
- How much other positions attended to it
- How the loss changes with respect to output

### Backward Through Softmax

```
Given: ∂Loss/∂attention_weights

Softmax derivative:
  ∂attention[i,j]/∂score[i,k] = attention[i,j] × (δ[j,k] - attention[i,k])

Where δ[j,k] = 1 if j==k, else 0
```

**Key property:** Gradients are proportional to attention weights themselves, which are bounded in [0,1], so gradients don't explode.

### Backward Through Scaling

```
∂Loss/∂scores = (1/√d_k) × ∂Loss/∂scaled_scores
```

Simple multiplication by constant.

### Backward Through Q @ K^T

```
Given: ∂Loss/∂scores

Compute:
  ∂Loss/∂Q = ∂Loss/∂scores @ K
  ∂Loss/∂K = (∂Loss/∂scores)^T @ Q
```

**Matrix multiplication chain rule.**

### Backward Through Projections

```
Given: ∂Loss/∂Q, ∂Loss/∂K, ∂Loss/∂V

Compute:
  ∂Loss/∂W_q = X^T @ ∂Loss/∂Q
  ∂Loss/∂W_k = X^T @ ∂Loss/∂K
  ∂Loss/∂W_v = X^T @ ∂Loss/∂V
  
  ∂Loss/∂X = ∂Loss/∂Q @ W_q^T + ∂Loss/∂K @ W_k^T + ∂Loss/∂V @ W_v^T
```

**Three paths for gradients to flow back to input!**

This is important: gradients can flow through Q, K, or V paths, making learning robust.

---

## Common Pitfalls and Debugging

### Pitfall 1: Uniform Attention

**Symptom:**
```
All attention weights are roughly equal:
attention[i] ≈ [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
```

**Causes:**
- Model hasn't learned yet (early training)
- Learning rate too low
- Scores are all very similar (Q and K not learning useful patterns)
- Softmax temperature effectively too high

**What it means:** Model isn't learning to distinguish relevant from irrelevant positions.

### Pitfall 2: Over-Focused Attention

**Symptom:**
```
Attention is nearly one-hot:
attention[i] ≈ [0.0, 0.0, 0.98, 0.0, 0.02, 0.0, 0.0, 0.0]
```

**Causes:**
- Scores too large (scaling issue)
- Softmax temperature effectively too low
- Model overfitting to training examples

**What it means:** Model ignoring potentially useful context from other positions.

### Pitfall 3: Gradient Vanishing

**Symptom:**
```
W_q, W_k, W_v stop updating (gradients ≈ 0)
```

**Causes:**
- Softmax saturation (one attention weight ≈ 1, others ≈ 0)
- Loss not sensitive to attention changes
- Dead neurons in later layers

**Solution:**
- Check attention weight distribution
- Verify loss is decreasing
- Use residual connections (Stage 4)

### Pitfall 4: NaN Values

**Symptom:**
```
Output contains NaN (not a number)
```

**Causes:**
- Numerical overflow in exp() during softmax
- Division by zero
- Invalid gradients

**Solution:**
- Ensure scores are properly scaled
- Clip extreme values before softmax
- Check for NaN in input data

---

## Attention Patterns: What Good Attention Looks Like

### For "Max(1,6,2)" - Well-Trained Model

**Position 0 ("Max"):**
```
Should attend to:
  - Itself (operation type): ~0.25
  - All three digits (2, 4, 6): ~0.20 each
  - Minimal to syntax: ~0.05 each

Good pattern:
  [0.28, 0.05, 0.22, 0.04, 0.21, 0.04, 0.19, 0.05]
   ↑         ↑              ↑              ↑
  Max       1              6              2
```

**Position 4 ("6"):**
```
Should attend to:
  - Operation "Max" (what to do): ~0.20
  - Itself (preserve value): ~0.20
  - Other digits (comparison): ~0.15 each
  - Minimal to syntax: ~0.05 each

Good pattern:
  [0.24, 0.06, 0.16, 0.05, 0.21, 0.05, 0.14, 0.05]
   ↑         ↑              ↑              ↑
  Max       1             self            2
```

### Bad Attention Patterns (Early Training)

**Uniform (not learning):**
```
[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
```

**Self-only (ignoring context):**
```
[0.02, 0.02, 0.02, 0.02, 0.88, 0.02, 0.02, 0.02]
```

**Random (confused):**
```
[0.23, 0.02, 0.31, 0.08, 0.12, 0.19, 0.03, 0.02]
```

---

## Computational Complexity

### Time Complexity

For sequence length S and dimension d_model:

**Q, K, V projections:**
- Each: O(S × d_model × d_k)
- For d_k = 16: O(8 × 64 × 16) = O(8,192)
- All three: O(3 × 8,192) = O(24,576)

**Attention scores (Q @ K^T):**
- O(S^2 × d_k)
- O(8^2 × 16) = O(1,024)

**Weighted sum (attention @ V):**
- O(S^2 × d_k)
- O(8^2 × 16) = O(1,024)

**Total:** O(S × d_model × d_k + S^2 × d_k)

**Key bottleneck:** The S^2 term!

For our toy problem (S=8): Negligible
For real LLMs (S=2048): S^2 = 4,194,304 → Major bottleneck!

This quadratic complexity with sequence length is why researchers develop efficient attention variants (sparse attention, linear attention, etc.).

### Space Complexity

**Attention weights storage:**
- O(S^2) = O(8^2) = O(64)
- For S=2048: 4MB per layer per batch element!

**Intermediate Q, K, V:**
- O(3 × S × d_k) = O(3 × 8 × 16) = O(384)

**For our model:** Tiny overhead
**For large models:** Memory is often the limiting factor

---

## Comparison to Other Mechanisms

### Attention vs RNNs

**RNN (Recurrent Neural Network):**
```
h_0 = f(x_0)
h_1 = f(h_0, x_1)
h_2 = f(h_1, x_2)
...
h_7 = f(h_6, x_7)
```

**Problems:**
- Sequential: Must process position-by-position
- Information bottleneck: h_t must compress all history
- Long-range dependencies: Information from h_0 must pass through 7 steps to reach h_7
- Not parallelizable: Can't compute h_7 until h_6 is done

**Attention:**
```
output_0 = weighted_sum(all positions)
output_1 = weighted_sum(all positions)
...
output_7 = weighted_sum(all positions)
```

**Advantages:**
- Parallel: All outputs computed simultaneously
- Direct connections: Position 0 can directly attend to position 7
- No bottleneck: Each position independently gathers what it needs
- Fully parallelizable: All computations can run in parallel on GPU

### Attention vs CNNs

**CNN (Convolutional Neural Network):**
```
Layer 1: Receptive field = 3 positions
Layer 2: Receptive field = 5 positions
Layer 3: Receptive field = 7 positions
```

**Problems:**
- Limited receptive field: Need many layers to see full sequence
- Fixed patterns: Convolution kernel is position-independent
- Hierarchical: Must build up understanding layer by layer

**Attention:**
```
Layer 1: Sees all 8 positions immediately
```

**Advantages:**
- Global receptive field: Sees everything from the start
- Dynamic patterns: Attention weights change based on input
- Direct: Don't need hierarchy to see distant positions

---

## Visualizing Attention

### Heatmap Representation

For a single attention head, we can plot the 8×8 attention matrix:

```
          Max  (   1   ,   6   ,   2   )
    Max [ ██  ░   ░   ░   █   ░   ░   ░  ] 0.28 to Max, 0.21 to 6
    (   [ ░   ██  ░   ░   ░   ░   ░   █  ] Self + closing paren
    1   [ ░   ░   ██  ░   █   ░   █   ░  ] Self + other numbers
    ,   [ ░   ░   ░   ██  ░   ░   ░   ░  ] Mostly self
    6   [ █   ░   █   ░   ██  ░   █   ░  ] Max + numbers
    ,   [ ░   ░   ░   ██  ░   ░   ░   ░  ] Mostly self
    2   [ ░   ░   █   ░   █   ░   ██  ░  ] Self + other numbers
    )   [ ░   █   ░   ░   ░   ░   ░   ██ ] Self + opening paren

Legend: ██ = 0.20-0.30, █ = 0.15-0.20, ░ = 0.05-0.15
```

### Graph Representation

Each token is a node, attention weights are edge weights:

```
         ┌─────────┐
         │   Max   │
         └────┬────┘
            / │ \
      0.22 /  │  \ 0.19
          /   │   \
         /0.21│    \
   ┌───▼─┐ ┌─▼──┐ ┌─▼──┐
   │  1  │ │ 6  │ │ 2  │
   └─────┘ └────┘ └────┘
       ▲     ▲     ▲
        \    |    /
      0.16\ │  /0.14
           \│ /0.21
        ┌───▼──────┐
        │    6     │
        └──────────┘
```

This shows how "Max" attends to all numbers, and "6" attends back to Max and other numbers.

---

## Connecting to Stage 3B (Preview)

Single-head attention is powerful, but limited:
- Only one 16-dimensional "view" of the data
- Must compress all relationships into one attention pattern
- Can't simultaneously learn different types of relationships

**Stage 3B (Multi-Head Attention)** will solve this by:
- Running 4 parallel attention heads
- Each with different learned weights (W_q, W_k, W_v)
- Each learning different aspects: operations, arguments, values, syntax
- Combining all 4 heads to get back to 64 dimensions

After Stage 3B, we'll have:
- Input: (8 × 64) 
- Output: (8 × 64) — same shape, but context-aware!
- Each position knows about all relevant other positions

---

## Key Takeaways

✓ **Attention computes relevance between all pairs of positions**

✓ **Q (Query) represents "what am I looking for?"**

✓ **K (Key) represents "what information do I have?"**

✓ **V (Value) represents "what information do I give?"**

✓ **Dot product (Q·K) measures relevance/similarity**

✓ **Scaling by √d_k prevents softmax saturation**

✓ **Softmax converts scores to probability distributions**

✓ **Weighted sum gathers information from relevant positions**

✓ **The mechanism is fully differentiable and learned end-to-end**

✓ **Each position becomes context-aware by attending to others**

✓ **Complexity is O(S²) — quadratic in sequence length**

---

## Understanding Check Questions

### Basic Conceptual Understanding

1. **Explain in your own words what the Query, Key, and Value represent. Use a real-world analogy (library, database, search engine, etc.).**

2. **Why do we need three separate projections (Q, K, V) instead of just using the input embeddings directly?**

3. **Position 4 ("6") attends to position 0 ("Max") with weight 0.148. What does this mean in concrete terms for the representation at position 4?**

4. **If we removed the softmax and just used raw scores as weights, what problems would occur? Give at least two specific issues.**

5. **Why is self-attention (diagonal values) typically high? Is this good or bad?**

### Mathematical Understanding

6. **Calculate the attention score by hand:**
   ```
   Q[2] = [1.0, 2.0, -1.0, 0.5]
   K[5] = [0.5, 1.0, -0.5, 1.0]
   d_k = 4
   
   a) Compute the dot product Q[2] · K[5]
   b) Compute the scaled score
   ```

7. **Given scaled scores for position 3:**
   ```
   scores[3] = [0.5, 1.0, 0.2, 0.8, 1.2, 0.3, 0.7, 0.4]
   
   a) Compute exp(score) for each value
   b) Compute the sum of all exp(score)
   c) Compute the attention weights after softmax
   d) Verify they sum to 1.0
   ```

8. **Why do we scale by √d_k specifically? What would happen if we scaled by d_k instead? Or didn't scale at all?**

9. **Given attention weights and Value vectors (first 2 dimensions only):**
    ```
    attention[4] = [0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1]
    
    V[0] = [1.0, 2.0, ...]
    V[1] = [0.5, 1.5, ...]
    V[2] = [2.0, 1.0, ...]
    V[3] = [0.5, 0.5, ...]
    V[4] = [1.5, 2.5, ...]
    V[5] = [1.0, 1.0, ...]
    V[6] = [0.5, 1.5, ...]
    V[7] = [1.0, 0.5, ...]
    
    Calculate output[4] for dimensions 0 and 1 by hand.
    ```

10. **For d_model = 64 and d_k = 16, sequence length S = 8:**
    - What is the shape of Q?
    - What is the shape of K^T?
    - What is the shape of the scores matrix?
    - What is the shape of the attention weights?
    - What is the shape of the final output?

### Practical Understanding

11. **In our problem Max(1,6,2), what attention pattern would you expect for:**
    - Position 0 ("Max") attending to other positions?
    - Position 2 ("1") attending to other positions?
    - Position 7 (")") attending to other positions?

12. **If attention weights were perfectly uniform [0.125, 0.125, ..., 0.125], what would this mean? Why is this considered "bad" attention?**

13. **Consider First(5, 3, 8). What attention pattern would you expect for:**
    - The "First" token?
    - The "5" token?
    
    How would this differ from the Max operation?

14. **Suppose all Value vectors were identical: V[0] = V[1] = ... = V[7]. What would happen to the output? Would attention still be useful?**

15. **Why doesn't the attention matrix need to be symmetric? Give an example where scores[i,j] ≠ scores[j,i] makes sense.**

### Computational Understanding

16. **Count the parameters in single-head attention:**
    - W_q dimensions and parameter count
    - W_k dimensions and parameter count
    - W_v dimensions and parameter count
    - Total parameters
    
    Show your work.

17. **Estimate the computational cost:**
    - FLOPs for Q, K, V projections (S=8, d_model=64, d_k=16)
    - FLOPs for computing scores Q @ K^T
    - FLOPs for weighted sum attention @ V
    - Which operation is the bottleneck for very long sequences?

18. **During the first epoch (random weights), describe what the attention patterns would look like and why. How do they evolve during training?**

### Advanced Understanding

19. **The diagonal of the attention matrix (self-attention) is often high. Explain why this makes sense:**
    - Mathematically (in terms of Q and K)
    - Intuitively (what does it mean for understanding?)

20. **Prove that the attention weights for any position sum to exactly 1.0 after softmax, regardless of the input scores.**

21. **Explain the gradient flow through the softmax operation. Why doesn't softmax cause vanishing gradients like the sigmoid activation function can?**

22. **If we used different scaling factors:**
    - softmax(scores / 2) — What effect?
    - softmax(scores / 10) — What effect?
    - softmax(scores × 2) — What effect?
    
    When might each be useful?

23. **Compare the memory requirements:**
    - Storing Q, K, V matrices
    - Storing attention weights
    - For S=8 vs S=2048, how does memory scale?

24. **Why is attention O(S²) in computational complexity? Show the analysis for:**
    - Computing scores: Q @ K^T
    - Computing weighted sum: attention @ V

25. **Design a modified attention mechanism that only allows each position to attend to its immediate neighbors (positions i-1, i, i+1). How would you modify:**
    - The scores computation?
    - The softmax operation?
    - What would be the new computational complexity?

### Interpretability and Analysis

26. **Given this attention pattern for the "Max" token:**
    ```
    [0.25, 0.05, 0.22, 0.05, 0.21, 0.05, 0.20, 0.02]
    ```
    Interpret what the model has learned. Is this good attention for the Max operation? Why or why not?

27. **Suppose you observe these attention patterns at different training stages:**
    
    Epoch 1: [0.13, 0.12, 0.13, 0.12, 0.13, 0.12, 0.13, 0.12]
    Epoch 10: [0.18, 0.09, 0.16, 0.08, 0.19, 0.08, 0.15, 0.07]
    Epoch 100: [0.24, 0.06, 0.22, 0.05, 0.21, 0.05, 0.19, 0.04]
    
    What story does this tell about learning?

28. **How would you expect attention patterns to differ between:**
    - First(5, 3, 8)
    - Max(5, 3, 8)
    - Min(5, 3, 8)
    
    Specifically, what would position 0 attend to in each case?

29. **If you observed that position 4 ("6") attends equally to all positions including commas and parentheses, what would this suggest about the model's learning?**

30. **Design a visualization method to understand what attention is doing. What would you plot? What insights would you look for?**

### Deep Dive Challenges

31. **The attention mechanism can be viewed as a differentiable key-value memory. Explain this perspective:**
    - What are the "keys"?
    - What are the "queries"?
    - What are the "values"?
    - How does "lookup" work?
    - Why is it "soft" rather than "hard"?

32. **Attention has been called "permutation equivariant." What does this mean? If we shuffle the input positions, what happens to the output?**

33. **Prove or disprove: The attention operation increases the "mutual information" between positions. If position 4's output contains information from position 0, has information increased or been redistributed?**

34. **Consider a pathological case where Q @ K^T produces a matrix of all zeros. What would the attention weights be? What would the output be? Is this a problem?**

35. **The paper "Attention Is All You Need" claimed attention could replace recurrence. For the task of "copy the input," show mathematically how attention can learn a perfect copy operation.**

---

## Experiments to Try

If you implement this in code, here are valuable experiments:

**1. Attention visualization:**
- Plot attention matrices as heatmaps at different training stages
- Watch how patterns emerge from uniform to focused
- Compare patterns for different operations (Max, Min, First)

**2. Ablation studies:**
- Remove scaling (÷ √d_k) — observe softmax saturation
- Use different d_k values — observe capacity changes
- Randomize V while keeping Q, K — observe information flow

**3. Weight analysis:**
- Examine learned W_q, W_k, W_v matrices
- Compute their singular values
- Measure rank and effective dimensionality

**4. Gradient monitoring:**
- Track gradient magnitudes through training
- Identify if any components have vanishing gradients
- Monitor attention weight entropy over time

**5. Attention pattern analysis:**
- Cluster positions by attention patterns
- Measure attention pattern similarity across examples
- Identify consistent patterns (e.g., "operations always attend to numbers")

---

## Common Misconceptions

**Misconception 1:** "Attention is like a weighted average"
- **Truth:** It IS a weighted average, but the weights are learned dynamically based on input content, not fixed.

**Misconception 2:** "Higher attention score means more important"
- **Truth:** After softmax, ALL weights contribute. Even 0.05 weight provides some information.

**Misconception 3:** "Attention looks at nearby positions"
- **Truth:** Attention can look anywhere! It's global, not local. The model learns where to look.

**Misconception 4:** "Self-attention (diagonal) is useless redundancy"
- **Truth:** Self-attention preserves original information while adding context. It's crucial!

**Misconception 5:** "Attention is computationally cheap"
- **Truth:** O(S²) complexity makes it expensive for long sequences (though fine for S=8).

**Misconception 6:** "Q, K, V must have different values"
- **Truth:** They're projections of the same input, so they're related, but serve different roles.

**Misconception 7:** "Softmax makes attention weights equal"
- **Truth:** Softmax preserves relative differences (exponentially amplifies them, actually).

---

## What's Next: Stage 3B Preview

Single-head attention gives us one 16-dimensional view of relationships. But our problem has multiple aspects:

**Syntactic relationships:**
- Matching parentheses
- Comma separators
- Expression structure

**Semantic relationships:**
- Operation type (Max, Min, First)
- Argument roles (1st, 2nd, 3rd argument)
- Functional relationships

**Value relationships:**
- Numerical comparisons
- Magnitude ordering
- Value selection

**One head can't learn all of these simultaneously!**

**Solution: Multi-Head Attention**
- Run 4 attention heads in parallel
- Each with different W_q, W_k, W_v (randomly initialized)
- Each learns to specialize in different patterns
- Combine all 4 heads to get full 64-dimensional output

**In Stage 3B, we'll explore:**
- How multi-head attention works mathematically
- Why heads naturally specialize (without being told to)
- How to combine heads effectively
- The role of the output projection W_o
- Real attention patterns from trained models

**Key insight:** Multi-head attention doesn't cost much more in parameters than single-head, but provides much more expressive power through specialization!

---

## Summary

In this lesson, we've built attention from the ground up:

1. **Problem:** Positions need to gather context from each other
2. **Solution:** Compute relevance (Q·K), normalize (softmax), gather (weighted sum of V)
3. **Mathematics:** Every step is differentiable and learnable
4. **Result:** Position representations become context-aware

**The attention formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Is actually a series of intuitive steps:**
1. Project input to queries, keys, values
2. Measure relevance with dot products
3. Scale to prevent saturation
4. Convert to probabilities with softmax
5. Gather information with weighted average

This mechanism is the foundation of modern transformers and the key innovation that made models like GPT, BERT, and others possible.

In Stage 3B, we'll see how running this mechanism multiple times in parallel (multi-head attention) dramatically increases its power!

---

## Further Reading

**Original paper:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- Section 3.2: Scaled Dot-Product Attention

**Key concepts to explore:**
- Self-attention vs cross-attention
- Masked attention for autoregressive models
- Relative positional attention
- Linear attention approximations
- Sparse attention patterns

**Visual resources:**
- Jay Alammar's "The Illustrated Transformer"
- 3Blue1Brown's "Attention in transformers, visually explained"
- TensorFlow's attention visualization tools

**Mathematical foundations:**
- Matrix calculus for backpropagation
- Information theory view of attention
- Kernel methods interpretation
    