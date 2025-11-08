# Transformer Deep Dive: Stage 1 & 2
## Embedding Layer and Positional Encoding

---

## Introduction: From Discrete Tokens to Continuous Space

When we feed text to a neural network, we face a fundamental problem: computers work with numbers, but language consists of discrete symbols (words, characters, tokens). How do we represent the word "Max" or the symbol "(" in a way that a neural network can process?

This is where **embeddings** and **positional encoding** come in. Together, they transform discrete tokens into rich, continuous representations that capture both meaning and position.

---

## STAGE 1: The Embedding Layer

### The Core Problem

Consider our example input: `Max(1,6,2)`

After tokenization, we have: `[15, 17, 3, 19, 8, 19, 4, 18]`

These are just integers—they have no inherent meaning to a neural network. The number 15 isn't "bigger" or "more important" than 3. We need a representation that:
- Captures semantic meaning
- Lives in continuous space (so gradients can flow)
- Can be learned from data
- Is the same dimensionality for all tokens

### What is an Embedding?

An embedding is a learned lookup table that maps each discrete token ID to a dense vector of real numbers.

**Mathematical Definition:**
```
E ∈ ℝ^(V × d_model)

Where:
  V = vocabulary size (20 tokens in our case)
  d_model = embedding dimension (64 in our case)
```

Think of it as a matrix where:
- Each **row** corresponds to one token in the vocabulary
- Each **column** represents one dimension of the embedding space
- Each **cell** contains a learned weight

### The Lookup Operation

For a token with ID `t`, the embedding is simply:
```
embedding(t) = E[t, :]
```

This extracts the t-th row from the embedding matrix.

**Concrete Example:**

```
Token: "Max" (ID = 15)

Embedding Matrix E:
         dim₀   dim₁   dim₂   dim₃  ...  dim₆₃
Token 0 [ 0.12, -0.34,  0.56, -0.23, ..., 0.45 ]  [PAD]
Token 1 [-0.23,  0.67, -0.12,  0.34, ..., -0.56 ]  [EOS]
...
Token 15[ 0.02, -0.15,  0.09, -0.08, ..., 0.07 ]  Max  ← This row
...
Token 19[ 0.34, -0.12,  0.45, -0.23, ..., 0.12 ]  ,

Result: embedding(15) = [0.02, -0.15, 0.09, -0.08, ..., 0.07]
```

### Processing a Sequence

For our input sequence `[15, 17, 3, 19, 8, 19, 4, 18]`, we look up each token:

```
Input Shape:  (8,)          # 8 token IDs
Output Shape: (8, 64)       # 8 vectors, each with 64 dimensions

Position 0: embedding(15) → [0.02, -0.15, 0.09, ..., 0.07]  "Max"
Position 1: embedding(17) → [-0.13, 0.29, -0.02, ..., 0.11]  "("
Position 2: embedding(3)  → [0.45, -0.23, 0.12, ..., -0.34]  "1"
Position 3: embedding(19) → [0.34, -0.12, 0.45, ..., 0.12]   ","
Position 4: embedding(8)  → [-0.11, 0.06, -0.02, ..., 0.09]  "6"
Position 5: embedding(19) → [0.34, -0.12, 0.45, ..., 0.12]   ","
Position 6: embedding(4)  → [0.23, -0.08, 0.15, ..., -0.19]  "2"
Position 7: embedding(18) → [0.19, -0.07, 0.23, ..., -0.11]  ")"
```

Notice that token 19 (comma) appears twice and gets the **exact same embedding** both times. The embedding is determined solely by the token ID, not its position.

### Mathematical Properties

**Dimensional Transformation:**
- Input: One-hot vector of size V (19 zeros and one 1)
- Output: Dense vector of size d_model (64 real numbers)  
- This is dimensional **expansion**: 20 → 64

**Important Note: Toy Model vs. Real LLMs**

In our toy example with 20 tokens, we expand from 20 to 64 dimensions. However, in real large language models, the opposite happens:

| Model | Vocabulary Size | Embedding Dimension | Transformation |
|-------|----------------|---------------------|----------------|
| Our Toy Model | 20 | 64 | Expansion (20 → 64) |
| LLaMA 3 | 128,256 | 4,096 | **Reduction** (128K → 4K) |
| DeepSeek-V3 | 128,000 | 7,168 | **Reduction** (128K → 7K) |

**Why the difference?**

1. **Semantic Redundancy in Natural Language:**
   - In natural language, many tokens are semantically similar:
     - "happy", "joyful", "delighted" (similar emotions)
     - "walk", "walked", "walking" (same verb, different forms)
     - "cat", "cats", "kitten" (related concepts)
   - With 128K tokens, there's massive redundancy that can be compressed
   - The model learns to map similar tokens to nearby points in embedding space

2. **Limited Redundancy in Our Toy Problem:**
   - Our 20 tokens are mostly distinct: operators (Max, Min), digits (0-9), syntax symbols
   - Little semantic overlap between "6" and "Max" or "(" and ","
   - We need MORE dimensions (not fewer) to give the model space to learn relationships

3. **Information Density vs. Dimensionality:**
   - One-hot (128K dims): Contains only log₂(128K) ≈ 17 bits of information ("which token?")
   - Dense embedding (4K dims): Encodes semantic meaning, relationships, context, learned concepts
   - Even with fewer dimensions, we gain far more representational power

**What Gets Learned:**

Regardless of whether we expand or reduce dimensions, the embedding layer learns to encode:
- Token identity
- Semantic relationships (e.g., "Max" and "Min" become similar vectors)
- Syntactic roles (e.g., digits cluster together)
- Abstract concepts that emerge during training (e.g., "larger than", "numeric value", "operator")

With only 20 dimensions, the model would have limited space to represent these complex relationships. With 64 dimensions, it has room to:
- Separate different types of tokens (operators, digits, syntax)
- Encode semantic similarities
- Learn task-specific features (e.g., which numbers are larger)
- Represent abstract concepts that aren't explicitly in the vocabulary

These learned concepts emerge during training through backpropagation, allowing the model to develop rich internal representations that go far beyond the surface-level token identities.

**Why Not One-Hot Encoding?**

One-hot encoding for "Max" (token 15) would be:
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

Problems with one-hot:
1. **Sparse:** Mostly zeros (inefficient)
2. **No similarity:** All tokens are equally distant from each other
3. **No learning:** Fixed representation that can't capture relationships
4. **No semantic meaning:** Can't represent that "6" and "7" are more similar than "6" and "Max"

Embeddings solve all these issues by using dense, continuous representations that can be learned from data.

### Initialization Strategies

Embeddings start with random values, but how we initialize matters:

**Xavier/Glorot Initialization (Common Choice):**
```
E[i, j] ~ Uniform(-√(1/d_model), √(1/d_model))

For d_model = 64:
E[i, j] ~ Uniform(-0.125, 0.125)
```

This ensures:
- Values aren't too large (would cause exploding gradients)
- Values aren't too small (would cause vanishing gradients)
- Variance is consistent across layers

**Initial Value Range:**
Typically in the range [-0.3, 0.3]

### How Embeddings Learn

During training, the embedding matrix E is updated via backpropagation:

```
∂Loss/∂E[t, :] = gradient from downstream layers

Update:
E[t, :] ← E[t, :] - learning_rate × ∂Loss/∂E[t, :]
```

**What Gets Learned:**

Over time, embeddings of similar tokens become similar:
- Digits (0-9) cluster together
- Operations (Max, Min) cluster together  
- Syntax tokens ( ) , form another cluster

**Geometric Interpretation:**

The 64 dimensions create a 64-dimensional space where:
- Distance captures similarity
- Direction captures semantic relationships
- Clusters emerge naturally

Example after training:
```
embedding("Max") ≈ embedding("Min")     # Both operations
embedding("6") ≈ embedding("7")         # Both digits
embedding("(") ≈ embedding(")")         # Both syntax
```

### Value Ranges During Training

**Beginning (Random):**
- Mean ≈ 0
- Standard deviation ≈ 0.15
- Range: [-0.3, 0.3]

**After 10 epochs:**
- Mean ≈ 0 (gradients push toward meaningful directions)
- Standard deviation ≈ 0.5
- Range: [-1.5, 1.5]

**After 50 epochs (Converged):**
- Mean ≈ 0
- Standard deviation ≈ 0.8
- Range: [-2.5, 2.5]

The range expands because the model learns to use the full representational capacity.

### Why 64 Dimensions?

This is a hyperparameter choice. Trade-offs:

**Fewer dimensions (e.g., 16):**
- ✓ Faster training
- ✓ Less memory
- ✗ Less representational power
- ✗ Harder to separate distinct concepts

**More dimensions (e.g., 256):**
- ✓ More representational power
- ✓ Can capture subtle distinctions
- ✗ Slower training
- ✗ Risk of overfitting on small datasets

For our toy problem with 20 tokens, 64 is generous. Real LLMs use:
- GPT-2: 768 dimensions
- GPT-3: 12,288 dimensions
- But they also have vocabularies of 50,000+ tokens

---

## STAGE 2: Positional Encoding

### The Problem: Attention is Permutation Invariant

The attention mechanism (which we'll see in Stage 3) computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

This operation is **invariant to the order** of inputs. If we shuffle the sequence, we get the same result (just shuffled).

**Example:**
- `Max(1,6,2)` → embeddings → attention
- `Max(6,2,1)` → embeddings → attention

Without positional information, these would be processed identically! But clearly:
- `First(1,6,2)` should output `1`
- `First(6,2,1)` should output `6`

Position matters crucially for our task.

### The Solution: Add Position Information

We add a position-dependent pattern to each embedding:
```
final_embedding = token_embedding + positional_encoding
```

The positional encoding must:
1. Be unique for each position
2. Generalize to sequences longer than training
3. Maintain consistent distances (position i and i+k should have similar patterns)
4. Not overpower the token embedding

### Sinusoidal Positional Encoding

The original Transformer paper uses sine and cosine functions:

**Mathematical Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in sequence (0, 1, 2, ..., 7)
  i = dimension index (0, 1, 2, ..., 31)
  d_model = 64 (embedding dimension)
```

This means:
- **Even dimensions** (0, 2, 4, ..., 62): Use sine
- **Odd dimensions** (1, 3, 5, ..., 63): Use cosine

### Breaking Down the Formula

**The wavelength factor:** `10000^(2i/d_model)`

For dimension i:
```
i = 0:  10000^(0/64)    = 1
i = 1:  10000^(2/64)    ≈ 1.17
i = 2:  10000^(4/64)    ≈ 1.37
i = 10: 10000^(20/64)   ≈ 3.98
i = 31: 10000^(62/64)   ≈ 6812
```

**Interpretation:**
- Low dimensions (i near 0): Fast oscillation (wavelength ~ 2π)
- High dimensions (i near 31): Slow oscillation (wavelength ~ 2π × 6812)

This creates a "binary encoding in continuous space"—different dimensions change at different rates as position increases.

### Computing Positional Encodings

**For position 0 (first token "Max"):**
```
PE(0, 0) = sin(0 / 1)         = sin(0) = 0.0
PE(0, 1) = cos(0 / 1)         = cos(0) = 1.0
PE(0, 2) = sin(0 / 1.17)      = sin(0) = 0.0
PE(0, 3) = cos(0 / 1.17)      = cos(0) = 1.0
...
PE(0, 62) = sin(0 / 6812)     = sin(0) = 0.0
PE(0, 63) = cos(0 / 6812)     = cos(0) = 1.0

Result: PE[0] ≈ [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, ..., 0.0, 1.0]
```

**For position 1 (second token "("):**
```
PE(1, 0) = sin(1 / 1)         = sin(1.0)    ≈ 0.841
PE(1, 1) = cos(1 / 1)         = cos(1.0)    ≈ 0.540
PE(1, 2) = sin(1 / 1.17)      = sin(0.855)  ≈ 0.755
PE(1, 3) = cos(1 / 1.17)      = cos(0.855)  ≈ 0.656
...
PE(1, 62) = sin(1 / 6812)     = sin(0.0001) ≈ 0.0001
PE(1, 63) = cos(1 / 6812)     = cos(0.0001) ≈ 1.0

Result: PE[1] ≈ [0.841, 0.540, 0.755, 0.656, ..., 0.0001, 1.0]
```

**For position 4 (fifth token "6"):**
```
PE(4, 0) = sin(4 / 1)         = sin(4.0)    ≈ -0.757
PE(4, 1) = cos(4 / 1)         = cos(4.0)    ≈ -0.654
PE(4, 2) = sin(4 / 1.17)      = sin(3.42)   ≈ -0.266
PE(4, 3) = cos(4 / 1.17)      = cos(3.42)   ≈ -0.964
...
PE(4, 62) = sin(4 / 6812)     = sin(0.0006) ≈ 0.0006
PE(4, 63) = cos(4 / 6812)     = cos(0.0006) ≈ 1.0

Result: PE[4] ≈ [-0.757, -0.654, -0.266, -0.964, ..., 0.0006, 1.0]
```

### Pattern Analysis

Notice the patterns:

**Low dimensions (fast oscillation):**
- Position 0: [0.0, 1.0]
- Position 1: [0.841, 0.540]
- Position 2: [0.909, -0.416]
- Position 3: [0.141, -0.990]
- Position 4: [-0.757, -0.654]

These change dramatically with each position.

**High dimensions (slow oscillation):**
- Position 0: [0.0, 1.0]
- Position 1: [0.0001, 1.0]
- Position 2: [0.0003, 0.9999]
- Position 3: [0.0004, 0.9999]
- Position 4: [0.0006, 0.9999]

These change very slowly—almost constant for nearby positions.

### Combining with Token Embeddings

**Element-wise addition:**
```
final[pos, dim] = embedding[pos, dim] + PE[pos, dim]
```

**For position 4 (token "6"):**
```
Token embedding:     [-0.11,  0.06, -0.02, ..., 0.09]
Positional encoding: [-0.76, -0.65, -0.27, ..., 1.00]
                     ─────────────────────────────────
Final embedding:     [-0.87, -0.59, -0.29, ..., 1.09]
```

**Complete sequence after Stage 2:**
```
Position 0 ("Max"):  [0.02 + 0.00,  -0.15 + 1.00,  ..., 0.07 + 1.00]
Position 1 ("("):    [-0.13 + 0.84, 0.29 + 0.54,   ..., 0.11 + 1.00]
Position 2 ("1"):    [0.45 + 0.91,  -0.23 + (-0.42),..., -0.34 + 1.00]
Position 3 (","):    [0.34 + 0.14,  -0.12 + (-0.99),..., 0.12 + 1.00]
Position 4 ("6"):    [-0.11 + (-0.76), 0.06 + (-0.65),..., 0.09 + 1.00]
Position 5 (","):    [0.34 + (-0.96), -0.12 + 0.28, ..., 0.12 + 1.00]
Position 6 ("2"):    [0.23 + (-0.28), -0.08 + 0.96, ..., -0.19 + 1.00]
Position 7 (")"):    [0.19 + 0.66,  -0.07 + 0.75,   ..., -0.11 + 1.00]
```

### Why Sinusoidal Functions?

**Advantages:**

1. **Deterministic:** No parameters to learn, so no risk of overfitting
2. **Unbounded:** Works for any sequence length (even longer than training)
3. **Relative positions:** The encoding for position p+k can be expressed as a linear function of the encoding for position p
4. **Smooth:** Changes gradually between adjacent positions
5. **Bounded:** All values in [-1, 1]

**Mathematical Property (Relative Position):**
```
PE(pos + k) can be computed as a linear combination of PE(pos)
```

This allows the model to learn to attend to relative positions (e.g., "three positions ahead") rather than just absolute positions.

**Alternative: Learned Positional Embeddings**

Instead of fixed sinusoidal, we could learn position embeddings just like token embeddings:
```
PE ∈ ℝ^(max_seq_len × d_model)  # Learned parameters
```

**Trade-offs:**
- ✓ Can adapt to specific tasks
- ✓ Might capture position better for this specific problem
- ✗ Cannot generalize beyond max_seq_len
- ✗ More parameters to learn (overfitting risk)

For our toy problem, sinusoidal is preferred for its generalization.

### Value Ranges After Stage 2

**Token embeddings:** ~[-0.3, 0.3] initially
**Positional encodings:** exactly [-1, 1] always
**Combined:** ~[-1.3, 1.3] initially, grows to ~[-3, 3] after training

The positional encoding is significant but doesn't dominate. Both position and token identity contribute to the final representation.

### Visual Intuition

Imagine each position as a unique "fingerprint":
- Position 0: A specific pattern of 64 numbers
- Position 1: A slightly different pattern
- Position 2: Another unique pattern
...

These fingerprints are added to the token embeddings, so the model sees:
- "This is token 'Max' at position 0"
- "This is token '6' at position 4"

The attention mechanism (Stage 3) can then use both **what** the token is and **where** it is.

---

## Why Both Stages Are Essential

**Stage 1 (Embeddings):** Answers "**WHAT** is this token?"
- Transforms discrete IDs into continuous vectors
- Captures semantic similarity
- Learned from data

**Stage 2 (Positional Encoding):** Answers "**WHERE** is this token?"
- Injects position information
- Breaks permutation invariance
- Fixed, not learned

**Together:** "This is token X at position Y"

Without embeddings: The model has no representation to work with.

Without positional encoding: The model cannot distinguish `First(1,6,2)` from `First(6,2,1)`.

---

## Understanding Check Questions

### Conceptual Understanding

1. **Why can't we use one-hot encodings instead of embeddings?** Describe at least three fundamental problems.

2. **The embedding matrix E has shape (20, 64). After looking up 8 tokens, what is the output shape and why?**

3. **Explain in your own words why transformers need positional encoding. Give a concrete example from our problem domain.**

4. **Why are positional encodings added to embeddings rather than concatenated?**

5. **What would happen if we used only learned positional embeddings trained on sequences up to length 10, and then tried to process a sequence of length 15?**

### Mathematical Understanding

6. **Calculate PE(2, 4) by hand:**
   - What is the wavelength factor for dimension 4?
   - Should we use sine or cosine?
   - What is the final value?

7. **For position 0, what is the pattern of the positional encoding across all 64 dimensions? Why does this pattern occur?**

8. **Given two embeddings after positional encoding:**
   - Position 3: [0.5, -0.3, 0.8, ..., 0.2]
   - Position 5: [0.5, -0.3, 0.8, ..., 0.2]
   
   Are these the same token at different positions or different tokens at the same position? Explain your reasoning.

9. **If we scale all embeddings by multiplying by 10 (E → 10E), but keep positional encodings the same, what problem would this create?**

10. **Why do low dimensions of positional encoding oscillate faster than high dimensions? What purpose does this serve?**

### Practical Understanding

11. **Token 19 (comma) appears at positions 3 and 5. Will these have:**
    - The same token embedding?
    - The same positional encoding?
    - The same final embedding after Stage 2?

12. **During training, which parameters are updated:**
    - Token embeddings?
    - Positional encodings (sinusoidal)?
    - Both?
    - Neither?

13. **Estimate the number of parameters in the embedding layer for our model. Show your calculation.**

14. **After training, we expect embeddings for "Max" and "Min" to be similar. True or false? Explain why or why not.**

15. **The positional encoding at position 100 (which is longer than our training sequences) will be:**
    - Undefined (will cause an error)
    - Random noise
    - A valid, unique pattern following the same formula
    - Identical to position 0

### Advanced Understanding

16. **Prove that sinusoidal positional encodings allow the model to learn relative positions. Hint: Show that PE(pos+k) can be expressed as a linear combination of PE(pos) using trigonometric identities.**

17. **Design an alternative positional encoding scheme that would work for our problem. What properties must it have?**

18. **The embedding dimension is 64, but our vocabulary size is only 20. Is this wasteful? Justify your answer considering both the current task and potential extensions.**

19. **Explain why the range of values in embeddings grows during training (from [-0.3, 0.3] to [-2.5, 2.5]). What does this tell us about what the model is learning?**

20. **If we removed positional encoding entirely, would the model still be able to learn anything useful for our problem? What could it learn, and what would it fail at?**

---

## Key Takeaways

✓ **Embeddings transform discrete symbols into continuous, learnable representations**

✓ **Positional encodings inject order information into the permutation-invariant attention mechanism**

✓ **The combination of token embeddings and positional encodings gives the model both semantic and structural information**

✓ **Sinusoidal encodings generalize to arbitrary sequence lengths without additional parameters**

✓ **The 64-dimensional embedding space allows rich representations to emerge through training**

✓ **These two stages set the foundation for all subsequent transformer operations**

✓ **In toy models with small vocabularies, embeddings expand dimensions (20 → 64)**

✓ **In real LLMs with large vocabularies, embeddings reduce dimensions (128K → 4K-7K)**

✓ **The choice depends on semantic redundancy: natural language has high redundancy, toy problems have low redundancy**

---

## What's Next?

In **Stage 3**, we'll see how the Multi-Head Attention mechanism uses these rich, position-aware embeddings to allow each token to gather information from all other tokens in the sequence. This is where the transformer's power really begins to shine!

The attention mechanism will compute which positions to focus on (e.g., the "Max" token learning to attend to all three digits), using the Query, Key, and Value projections we'll explore in detail.
