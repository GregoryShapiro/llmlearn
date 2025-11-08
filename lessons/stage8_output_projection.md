# Transformer Deep Dive: Stage 8
## Output Projection: From Representations to Predictions

---

## Introduction: The Final Transformation

After Stages 1-7, our transformer has performed an incredible journey:
- **Stage 1-2**: Converted tokens to position-aware embeddings
- **Stage 3-6 (Block 1)**: Gathered context and transformed representations
- **Stage 7 (Block 2)**: Refined understanding with hierarchical patterns

Now we have a final representation for each position: shape (8 × 64). But we face a fundamental question: **How do we convert these abstract 64-dimensional vectors into actual predictions?**

For `Max(1,6,2)`, we need to predict that the answer is "6" (token ID 8 in our vocabulary). Stage 8 solves this by projecting from representation space (64 dimensions) to vocabulary space (20 dimensions), where each dimension corresponds to one possible token.

This is where the model makes its prediction.

---

## The Fundamental Problem: From Features to Tokens

Consider the final output from Block 2 after processing `Max(1,6,2)`:

```
Position 0: "Max"  → [0.722, -1.419, 0.913, -1.002, ..., 0.867]  (64 dims)
Position 1: "("    → [0.453, -0.891, 0.234, -0.567, ..., 0.432]  (64 dims)
Position 2: "1"    → [-0.234, 0.678, -0.456, 0.890, ..., -0.321]  (64 dims)
Position 3: ","    → [0.567, -0.234, 0.789, -0.123, ..., 0.654]  (64 dims)
Position 4: "6"    → [0.722, -1.419, 0.913, -1.002, ..., 0.867]  (64 dims)
Position 5: ","    → [0.345, -0.678, 0.234, -0.456, ..., 0.543]  (64 dims)
Position 6: "2"    → [-0.456, 0.890, -0.234, 0.678, ..., -0.234]  (64 dims)
Position 7: ")"    → [0.234, -0.567, 0.891, -0.234, ..., 0.456]  (64 dims)
```

Each position has a rich 64-dimensional representation encoding:
- What token it is
- Its position in the sequence
- Context from all other tokens
- Hierarchical features from both transformer blocks

**But we need to predict a single token:** the answer "6".

**The challenge:** How do we go from these abstract feature vectors to concrete token predictions?

---

## PART 1: Selecting the Representation

### Which Position Do We Use?

For our task (predicting the result of an operation), we need to choose which position's representation to use for the final prediction.

**Three common strategies:**

**Strategy 1: Last Position**
```python
final_repr = output[sequence_length - 1]
```
- Used in autoregressive models (GPT)
- Assumes the last position aggregates all information
- For `Max(1,6,2)`, use position 7 (")")

**Strategy 2: First Position (CLS token)**
```python
final_repr = output[0]
```
- Used in BERT-style models
- First position dedicated to classification
- For `Max(1,6,2)`, use position 0 ("Max")

**Strategy 3: Pooling (Average/Max)**
```python
final_repr = mean(output, axis=0)  # or max
```
- Aggregates information from all positions
- More robust but loses positional specificity

**For our model:** We'll use the **first position** (position 0) because:
1. It corresponds to the operation being performed
2. Multi-head attention has already aggregated information from all arguments
3. It's the natural "question" position that should contain the "answer"

**After selection:**
```
final_repr = output[0] = [0.722, -1.419, 0.913, -1.002, ..., 0.867]  (64 dims)
```

---

## PART 2: The Output Projection Layer

### The Transformation

We need to convert from 64 dimensions (feature space) to 20 dimensions (vocabulary space):

```
logits = final_repr @ W_output + b_output

Where:
  final_repr ∈ ℝ^64      (input representation)
  W_output ∈ ℝ^(64×20)   (learned weight matrix)
  b_output ∈ ℝ^20        (learned bias vector)
  logits ∈ ℝ^20          (output scores for each token)
```

**Why 20 dimensions?** Because our vocabulary has 20 tokens:
```
0: [PAD]    1: [EOS]    2: '0'    3: '1'    4: '2'
5: '3'      6: '4'      7: '5'    8: '6'    9: '7'
10: '8'     11: '9'     12: 'First'  13: 'Second'  14: 'Last'
15: 'Max'   16: 'Min'   17: '('   18: ')'   19: ','
```

Each dimension of the output corresponds to one token. The model learns which dimension to "activate" for each prediction.

---

### Step-by-Step Computation

**Step 1: Matrix Multiplication**

```
logits = final_repr @ W_output

Dimension check:
(1 × 64) @ (64 × 20) = (1 × 20)  ✓
```

**Concrete example:**

```
final_repr = [0.722, -1.419, 0.913, -1.002, 0.345, -0.678, 0.234, ..., 0.867]

W_output (showing first 4 columns):
         [PAD]  [EOS]  '0'    '1'
dim 0:   0.12   0.34  -0.23   0.45  ...
dim 1:  -0.45   0.67   0.89  -0.12  ...
dim 2:   0.34  -0.23   0.56   0.78  ...
dim 3:  -0.89   0.45  -0.12   0.34  ...
...
dim 63:  0.23  -0.56   0.34  -0.78  ...

For token 0 ([PAD]):
logits[0] = 0.722×0.12 + (-1.419)×(-0.45) + 0.913×0.34 + (-1.002)×(-0.89) + ...
         = 0.087 + 0.639 + 0.310 + 0.892 + ...
         = -2.453

For token 8 ('6'):
logits[8] = 0.722×(W[0,8]) + (-1.419)×(W[1,8]) + 0.913×(W[2,8]) + ...
         = 3.782
```

**Step 2: Add Bias**

```
logits = logits + b_output

b_output = [0.1, -0.05, 0.02, ..., -0.03]

Final logits:
logits[0] = -2.453 + 0.1 = -2.353
logits[8] = 3.782 + (-0.03) = 3.752
...
```

**Complete output (logits for all 20 tokens):**

```
Token ID  | Token    | Logit Value | Interpretation
----------|----------|-------------|----------------------------------
0         | [PAD]    | -2.35       | Very unlikely (padding)
1         | [EOS]    | -1.48       | Unlikely (not end of sequence)
2         | '0'      | -0.82       | Somewhat unlikely
3         | '1'      | -0.31       | Neutral
4         | '2'      | 1.23        | Possible (but not maximum)
5         | '3'      | 0.73        | Possible
6         | '4'      | 0.92        | Possible
7         | '5'      | 1.54        | Strong possibility
8         | '6'      | 3.75        | HIGHEST! Model's prediction ✓
9         | '7'      | 0.41        | Possible
10        | '8'      | 0.23        | Neutral
11        | '9'      | -0.08       | Neutral
12        | 'First'  | -1.92       | Very unlikely (wrong operation)
13        | 'Second' | -1.67       | Very unlikely
14        | 'Last'   | -1.43       | Very unlikely
15        | 'Max'    | -2.01       | Very unlikely (already used)
16        | 'Min'    | -1.78       | Very unlikely
17        | '('      | -2.34       | Very unlikely (syntax)
18        | ')'      | -2.12       | Very unlikely (syntax)
19        | ','      | -2.45       | Very unlikely (syntax)
```

**Key insight:** The highest logit is at position 8, which corresponds to token "6"—exactly the correct answer for `Max(1,6,2)`!

---

## PART 3: Understanding Logits

### What Are Logits?

**Logits** are **unnormalized log-probabilities**. They are raw scores before applying softmax.

**Key properties:**

1. **Unbounded:** Can be any real number (positive, negative, large, small)
2. **Relative, not absolute:** Only the differences matter
3. **Higher = more likely:** The model predicts the token with the highest logit

**Why not output probabilities directly?**

Because we'll apply **softmax** in Stage 9 to convert logits to probabilities:
```
probability[i] = exp(logits[i]) / Σⱼ exp(logits[j])
```

This two-step process (logits → softmax) has numerical stability benefits and cleaner gradient computation.

---

### What The Model Has Learned

**The weight matrix W_output encodes token semantics:**

Each row corresponds to one embedding dimension, and each column corresponds to one vocabulary token. During training, the model learns:

**For digit tokens (2-11):**
- Dimensions that correlate with numerical value
- Patterns that distinguish between digits

**For operation tokens (12-16):**
- Dimensions that recognize operation types
- Features that are anti-correlated (since operations shouldn't be answers)

**For syntax tokens (17-19):**
- Strong negative weights (syntax tokens are never answers)

**Example: Column for token '6' (index 8):**
```
W_output[:, 8] = [0.34, -0.89, 0.56, 0.23, -0.45, ..., 0.78]
                  ↑     ↑      ↑      ↑      ↑          ↑
                These weights are tuned to respond strongly when
                final_repr contains features indicating "the answer is 6"
```

**How training discovers this:**

During training, whenever the correct answer is "6":
1. Backpropagation increases weights in W_output[:, 8] that correlate with features in final_repr
2. Backpropagation decreases weights in other columns
3. Over many examples, W_output[:, 8] learns to recognize "answer is 6" patterns

---

## PART 4: Geometric Interpretation

### Representations as Points in Space

Think of each possible representation as a point in 64-dimensional space:

```
Representations after Block 2:
  When answer is '6': cluster in one region of space
  When answer is '1': cluster in different region
  When answer is '9': cluster in another region
```

**The output projection is a linear classifier:**

```
W_output defines 20 hyperplanes in 64D space
Each hyperplane separates one token from the rest

For token '6':
  logits[8] = w₈ᵀ · final_repr + b₈
  
This is a hyperplane equation!
  Decision boundary: w₈ᵀ · x + b₈ = 0
```

**Visualization (reduced to 2D):**

```
                              Representations that produce '6'
                                      ↓↓↓↓↓
                                    • • • •
                                   • • • • •
             Hyperplane for '6'   • • • • •
                    ↓            ─────────────
                    │           /
                    │          /
     • •            │         /        • • •  ← Representations
    • • •           │        /        • • • •    that produce '1'
   • • • •          │       /        • • • • •
    • • •           │      /          • • •
     • •            │     /            •
                    │    /
  Representations   │   /
  that produce '2'  │  /
                    │ /
                    │/
```

**What training does:**

1. **Block 1 and Block 2** learn to map input sequences to well-separated regions
2. **Output projection** learns hyperplanes to separate these regions
3. The combination creates **linearly separable representations**

This is why the final layer can be a simple linear projection—the hard work of creating separable representations is done by the transformer blocks!

---

## PART 5: Parameter Analysis

### Output Projection Parameters

**Weight Matrix: W_output**
```
Shape: (64 × 20) = 1,280 parameters
Interpretation: Each row is one embedding dimension
                Each column is one vocabulary token
```

**Bias Vector: b_output**
```
Shape: (20,) = 20 parameters
Interpretation: Per-token bias (prior probability)
```

**Total Stage 8 parameters: 1,280 + 20 = 1,300**

**Proportion of total model:**
```
Output projection: 1,300 parameters
Total model: 102,016 parameters
Percentage: 1.3%
```

The output projection is tiny compared to the transformer blocks! This makes sense—the hard work is learning good representations, not the final classification.

---

### Understanding the Bias Term

**What does b_output represent?**

The bias gives each token a baseline score independent of the input:

```
logits[i] = (W_output[:, i])ᵀ · final_repr + b_output[i]
            └─────────────────────────────┘   └─────────┘
              Content-dependent               Prior
```

**Example:**

If token '6' appears frequently in training data:
- b_output[8] might be slightly positive (e.g., 0.1)
- This gives '6' a small baseline advantage

If token [PAD] should never be predicted:
- b_output[0] might be very negative (e.g., -3.0)
- This strongly discourages predicting padding

**After training, typical bias values:**
```
b_output = [
  -3.2,  # [PAD] - should never predict
  -2.1,  # [EOS] - rarely the answer
   0.1,  # '0'   - common digit
   0.2,  # '1'   - common digit
   0.3,  # '2'   - common digit
   ...
  -2.5,  # 'Max' - operations aren't answers
  -2.3,  # '('   - syntax isn't answers
]
```

---

## PART 6: Connection to Classification

### Output Projection as Multi-Class Classification

Stage 8 is essentially a **20-way classification** problem:

```
Input: final_repr (64 dimensions)
Output: Which of 20 tokens to predict
Method: Linear classifier (logistic regression)
```

**This is identical to the output layer of any classification neural network!**

**Comparison to standard classification:**

```
Image Classification          | Transformer Output
------------------------------|----------------------------
Input: CNN features (2048)    | Input: Transformer repr (64)
Output: Class logits (1000)   | Output: Token logits (20)
Layer: Linear(2048 → 1000)    | Layer: Linear(64 → 20)
Method: Softmax + cross-entropy | Method: Softmax + cross-entropy
```

The transformer is a sophisticated **feature extractor**, and Stage 8 is a simple classifier on top of those features.

---

### Why Linear Is Sufficient

**Question:** Why can a linear layer make complex predictions?

**Answer:** Because the transformer blocks have already done the hard work!

**Feature learning hierarchy:**
```
Stage 1-2: Basic token + position features
Stage 3-6 (Block 1): Context-aware features
Stage 7 (Block 2): Abstract reasoning features
Stage 8: Linear combination for prediction
```

By Stage 8, the representations are already **linearly separable**:
- All "answer is 6" sequences cluster together
- All "answer is 1" sequences cluster in a different region
- Etc.

A linear classifier is perfect for linearly separable data!

**Alternative architectures (not used here but possible):**
1. **Multi-layer classifier:** Add more fully-connected layers before output
2. **Nonlinear activation:** Add ReLU or tanh before final projection
3. **Larger projection:** Expand to intermediate size then contract

But empirically, a simple linear projection works excellently for transformers because the blocks create such good representations.

---

## PART 7: Prediction Example Walkthrough

Let's trace a complete prediction for `Max(1,6,2)`:

**Input sequence:**
```
Tokens: [Max, (, 1, ,, 6, ,, 2, )]
IDs:    [15, 17, 3, 19, 8, 19, 4, 18]
```

**After all transformer processing:**
```
Block 2 output[0] = [0.722, -1.419, 0.913, -1.002, ..., 0.867]
```

**Output projection computation:**

```
logits = output[0] @ W_output + b_output

Detailed calculation for each token:
─────────────────────────────────────────────────
Token '0' (index 2):
  = 0.722×W[0,2] + (-1.419)×W[1,2] + ... + 0.867×W[63,2] + b[2]
  = -0.82

Token '1' (index 3):
  = 0.722×W[0,3] + (-1.419)×W[1,3] + ... + 0.867×W[63,3] + b[3]
  = -0.31

Token '6' (index 8):
  = 0.722×W[0,8] + (-1.419)×W[1,8] + ... + 0.867×W[63,8] + b[8]
  = 3.75  ← HIGHEST!
─────────────────────────────────────────────────
```

**Prediction:**
```
predicted_token_id = argmax(logits) = 8
predicted_token = vocabulary[8] = '6'

CORRECT! ✓
```

**What if the input was `Max(1,3,2)`?**

The final representation would be different:
```
Block 2 output[0] = [0.543, -1.102, 0.776, -0.887, ..., 0.654]
                     ↑       ↑       ↑       ↑           ↑
                    Slightly different pattern
```

This would produce:
```
logits[3] = 2.84  ← Highest (token '3')
logits[8] = 1.42  ← Lower than before
```

Prediction: '3' ✓ Correct!

---

## PART 8: Training Dynamics

### How W_output Is Learned

**Initial state (random initialization):**
```
W_output ∈ ℝ^(64×20), initialized from 𝒩(0, 0.02)
b_output ∈ ℝ^20, initialized to zeros

Predictions are essentially random!
```

**During training:**

**Example: Training on `Max(5, 9, 3)` with correct answer '9' (token 11)**

**Step 1: Forward pass**
```
logits = final_repr @ W_output + b_output
logits[11] = 0.82  (token '9')
logits[7] = 1.45   (token '5') ← Wrong prediction!
```

**Step 2: Compute loss**
```
loss = -log(softmax(logits)[11])  (cross-entropy)
```

The loss is high because the model incorrectly predicted '5' instead of '9'.

**Step 3: Backpropagation**

Gradients flow backward:
```
∂loss/∂logits[11] = probability[11] - 1 ≈ 0.31 - 1 = -0.69
∂loss/∂logits[7] = probability[7] - 0 ≈ 0.52 - 0 = 0.52
```

This means:
- **Increase** logits[11] (correct answer)
- **Decrease** logits[7] (wrong answer)

**Step 4: Update W_output**

```
For column 11 (token '9'):
  ∂loss/∂W_output[:, 11] = final_repr × (-0.69)
  
  W_output[:, 11] -= learning_rate × final_repr × (-0.69)
  W_output[:, 11] += learning_rate × final_repr × 0.69
  
  This increases weights that correlate with "answer is 9"

For column 7 (token '5'):
  W_output[:, 7] -= learning_rate × final_repr × 0.52
  
  This decreases weights that incorrectly predicted '5'
```

**Over many training steps:**
- W_output[:, 9] learns to recognize patterns → "answer is 9"
- W_output[:, 5] learns to recognize patterns → "answer is 5"
- Etc. for all tokens

---

### Convergence Pattern

**Training progression:**

```
Epoch 0:  Loss = 3.00 (random guessing)
          Accuracy = 5% (chance level for 20 classes)

Epoch 10: Loss = 1.85
          Accuracy = 32%
          Model starts learning digit tokens

Epoch 50: Loss = 0.45
          Accuracy = 78%
          Model distinguishes operations well

Epoch 100: Loss = 0.12
          Accuracy = 96%
          Model predicts confidently and accurately

Epoch 200: Loss = 0.03
          Accuracy = 99.5%
          Near-perfect predictions
```

**What changes in W_output:**

```
Early training (Epoch 10):
  W_output columns are somewhat random
  Some structure emerges (digits vs operations)

Mid training (Epoch 50):
  Clear separation between digit/operation/syntax columns
  Digit columns develop value-ordering patterns

Late training (Epoch 200):
  Highly specialized columns
  Strong anti-correlation with wrong answers
  Confident, stable predictions
```

---

## PART 9: Comparison Across Architectures

### Output Layers in Different Models

**GPT (Autoregressive Language Model):**
```
Uses: Last position representation
Shape: (vocab_size,) e.g., (50257,)
Task: Predict next token
Output: W_output @ final_repr
```

**BERT (Masked Language Model):**
```
Uses: Masked position representation
Shape: (vocab_size,) e.g., (30522,)
Task: Predict masked token
Output: W_output @ masked_repr
```

**Our Model (Sequence Classification):**
```
Uses: First position representation
Shape: (vocab_size,) = (20,)
Task: Predict operation result
Output: W_output @ final_repr
```

**Key similarity:** All use **linear projection** from representation space to vocabulary space.

**Key difference:** Which representation to use (last, masked, first, pooled).

---

### Weight Tying

**Advanced technique:** In some models, W_output is tied to the embedding matrix:

```
W_output = W_embedding^T

Benefits:
  - Reduces parameters (only store one matrix)
  - Enforces symmetry (embedding and unembedding are related)
  - Sometimes improves generalization

Our model: Uses separate W_output (more flexible)
```

---

## PART 10: Complete Forward Pass Summary

Let's see the complete journey from input to logits:

```
INPUT: "Max(1,6,2)"

STAGE 1: Embedding
  [15, 17, 3, 19, 8, 19, 4, 18] → (8 × 64)

STAGE 2: Positional Encoding
  Add position information → (8 × 64)

STAGE 3-6: Transformer Block 1
  Multi-head attention + FFN → (8 × 64)

STAGE 7: Transformer Block 2
  Multi-head attention + FFN → (8 × 64)
  Output[0] = [0.722, -1.419, 0.913, ..., 0.867]

STAGE 8: Output Projection
  [0.722, -1.419, ..., 0.867] @ W_output + b → (20,)
  
  Logits = [-2.35, -1.48, -0.82, -0.31, 1.23, 0.73, 
            0.92, 1.54, 3.75, 0.41, 0.23, -0.08, 
            -1.92, -1.67, -1.43, -2.01, -1.78, 
            -2.34, -2.12, -2.45]

PREDICTION: argmax(logits) = 8 → Token '6'

NEXT (Stage 9): Softmax to get probabilities
NEXT (Stage 10): Compute loss and backpropagate
```

---

## Key Takeaways

✓ **Output projection transforms representations (64D) into vocabulary scores (20D)**

✓ **Logits are unnormalized scores; higher values = more likely predictions**

✓ **The projection layer is a simple linear classifier on top of rich features**

✓ **Each column of W_output learns to recognize patterns for one token**

✓ **The bias term b_output encodes prior probabilities for each token**

✓ **Linear projection is sufficient because transformer blocks create linearly separable representations**

✓ **The output layer constitutes only ~1.3% of model parameters**

✓ **Stage 8 is mathematically identical to multi-class logistic regression**

✓ **Training gradually specializes W_output columns for each vocabulary token**

✓ **The highest logit determines the model's prediction (before softmax)**

---

## Understanding Check Questions

### Conceptual Understanding

1. **Explain why we use a linear projection instead of a more complex multi-layer network for the output layer.**

2. **What is the difference between logits and probabilities? Why do we compute logits first, then apply softmax?**

3. **In our model, we use the first position (position 0) for prediction. Explain why this choice makes sense for our task and how it might differ for other tasks.**

4. **The bias term b_output has 20 parameters. Explain what each parameter represents and give an example of when b_output[i] should be very negative.**

5. **Why is the output projection layer so much smaller (1,300 parameters) compared to transformer blocks (49,728 parameters each)?**

6. **Describe what happens to W_output during training when the model incorrectly predicts '3' but the correct answer is '7'.**

### Mathematical Understanding

7. **Calculate the output logit for token 5 ('3') given:**
   ```
   final_repr = [0.5, -0.3, 0.8, -0.2] (simplified 4D)
   W_output[:, 5] = [0.6, -0.4, 0.2, 0.1]
   b_output[5] = 0.05
   ```

8. **Verify the parameter count: W_output has shape (64 × 20) and b_output has shape (20,). Show that the total is 1,300 parameters.**

9. **If logits = [-1.0, 2.0, 3.5, 1.0, -0.5], what is argmax(logits)? What token does this correspond to if token IDs match indices?**

10. **Given three representations that produce different answers:**
    ```
    repr_1 produces logits: [0.5, 2.0, 1.0]  (predicts token 1)
    repr_2 produces logits: [1.5, 1.0, 2.5]  (predicts token 2)
    repr_3 produces logits: [3.0, 0.5, 1.0]  (predicts token 0)
    ```
    **Explain why these representations must lie in different regions of the 64D space.**

11. **Compute the gradient ∂loss/∂W_output[:, 8] when:**
    - Target token is 8 ('6')
    - final_repr = [0.4, -0.6, 0.3, 0.1, ...]
    - Current probability for token 8 is 0.85
    (Hint: gradient = final_repr × (probability - 1))

12. **For two different inputs:**
    ```
    Input A: Max(5, 9, 2) → final_repr_A
    Input B: Max(5, 9, 3) → final_repr_B
    ```
    **Both should predict '9'. Explain why final_repr_A and final_repr_B should be similar but not identical.**

### Architectural Understanding

13. **Compare the output layer of our transformer to the output layer of a convolutional neural network for image classification. What are the similarities and differences?**

14. **Some models use "weight tying" where W_output = W_embedding^T. What are the advantages and disadvantages of this approach?**

15. **Suppose we wanted to predict multiple outputs (e.g., both the maximum and minimum). How would you modify Stage 8 to accommodate this?**

16. **Explain why transformers typically use the same output projection architecture regardless of vocabulary size (20 tokens vs 50,000 tokens).**

### Training Dynamics Understanding

17. **At initialization, W_output is random. Describe what the logits look like and what accuracy the model achieves.**

18. **During training, you observe that b_output[0] (for [PAD]) becomes very negative (-5.2). Explain why this happens and whether it's desirable.**

19. **After 50 epochs, you notice that W_output[:, 8] (for token '6') and W_output[:, 11] (for token '9') have similar patterns in their first 32 dimensions but differ significantly in the last 32 dimensions. What might this tell you about what features the model learned?**

20. **You observe that during training, the logit for the correct answer increases from 0.5 to 3.8, while logits for wrong answers decrease from 0.3 to -1.2. Explain why both changes contribute to better predictions.**

### Geometric and Representational Understanding

21. **In 2D space, sketch three clusters of points representing "answer is 1", "answer is 2", and "answer is 3". Draw the decision boundaries (hyperplanes) that the output projection learns.**

22. **Explain what it means for representations to be "linearly separable" and why this property makes a linear output layer sufficient.**

23. **If two different input sequences `Max(5,9,2)` and `Max(3,9,1)` both have answer '9', should their final representations be identical? Why or why not?**

24. **The output projection defines 20 hyperplanes in 64D space. Explain how these hyperplanes partition the space into regions, and what each region represents.**

25. **Consider the dot product: logits[i] = w_i^T · final_repr. Explain geometrically what it means when this dot product is large versus small.**

### Practical Understanding

26. **You deploy your model and notice it occasionally predicts syntax tokens like '(' or ')' as answers, even though they should never be answers. What does this suggest about the training data or the model architecture?**

27. **Suppose you want to add a new operation "Median(a,b,c)" to your vocabulary. What changes do you need to make to Stage 8? Can you reuse the trained W_output or must you retrain?**

28. **The model has trouble distinguishing between very similar operations like Max and Min. Looking only at Stage 8, what might be happening with W_output[:, 15] (Max) and W_output[:, 16] (Min)?**

29. **You notice that logits for digits vary in magnitude (some are 3.5, others are 1.2) even when the model is confident. Does this matter for prediction? Does this matter for training?**

30. **Design a diagnostic test to check whether Stage 8 is the bottleneck in your model's performance versus earlier stages.**

### Advanced Exploration Questions

31. **If you were to visualize W_output as a 64×20 heatmap, what patterns would you look for to understand what the model learned?**

32. **Compare single-task output projection (our model: 20 tokens) versus multi-task output projection (e.g., predicting operation type AND result). How would the architecture change?**

33. **Some language models use "adaptive softmax" which treats frequent and rare tokens differently. Explain why this might be beneficial and how it relates to our output projection.**

34. **Suppose you want the model to output a confidence score along with the prediction. Can you extract this from the logits? How?**

35. **Design an experiment to measure how much of the model's performance depends on the quality of representations (Stages 1-7) versus the quality of the output projection (Stage 8).**

---

## Deep Dive: Logits to Predictions Pipeline

### The Three Stages of Making Predictions

Even though we call this "Stage 8", the complete prediction pipeline actually spans three stages:

**Stage 8: Output Projection (this stage)**
```
Input: final_repr (64D)
Process: Linear transformation
Output: logits (20D, unnormalized)
```

**Stage 9: Softmax (next stage)**
```
Input: logits (20D)
Process: Exponential normalization
Output: probabilities (20D, sum to 1)
```

**Stage 10: Loss and Prediction (next stage)**
```
Input: probabilities (20D)
Process: Cross-entropy loss, argmax
Output: predicted_token, loss_value
```

**Why separate these stages?**

1. **Modularity**: Each stage has a clear mathematical purpose
2. **Numerical stability**: Separating logits and softmax allows for log-space computations
3. **Gradient flow**: Different gradient computations for each stage
4. **Interpretability**: Logits and probabilities provide different insights

---

## Deep Dive: Feature Learning in W_output

### What Patterns Emerge in W_output?

After training, if we analyze the weight matrix W_output, we can discover what features the model learned:

**Experiment: Clustering W_output Columns**

If we cluster the 20 columns of W_output, we find natural groupings:

```
Cluster 1: Digit tokens (2-11)
  W_output[:, 2] through W_output[:, 11]
  Similar patterns, gradual variation by value

Cluster 2: Operation tokens (12-16)
  W_output[:, 12] through W_output[:, 16]
  Distinct from digits, share negative correlations

Cluster 3: Syntax tokens (17-19)
  W_output[:, 17] through W_output[:, 19]
  Very negative overall, never predicted

Cluster 4: Special tokens (0-1)
  W_output[:, 0] and W_output[:, 1]
  Unique patterns for padding/end-of-sequence
```

**Visualization of structure:**

```
            Dim 0 Dim 1 Dim 2 ... Dim 63
Token '0':   0.2  -0.3   0.1  ...  0.4   ]
Token '1':   0.3  -0.2   0.2  ...  0.5   ] Similar
Token '2':   0.4  -0.1   0.3  ...  0.6   ] patterns
Token '3':   0.5   0.0   0.4  ...  0.7   ]
...
Token '9':   1.2   0.7   1.1  ...  1.4   ]

Token 'Max': -0.8  -1.2  -0.6  ... -0.9  ] Different
Token 'Min': -0.7  -1.1  -0.5  ... -0.8  ] pattern
```

**Interpreting dimensions:**

Some dimensions in W_output may specialize:
- **Dimensions 0-15**: Might encode "numerical value" (high for digit 9, low for digit 0)
- **Dimensions 16-31**: Might encode "is this a digit?" (high for digits, low for operations)
- **Dimensions 32-47**: Might encode "task-relevant token" (high for potential answers)
- **Dimensions 48-63**: Might encode specialized features discovered during training

---

## Deep Dive: Gradient Flow Through Stage 8

### How Gradients Update W_output

During backpropagation, gradients flow from the loss through the output projection:

**Forward pass:**
```
logits = final_repr @ W_output + b_output
```

**Backward pass:**
```
Given: ∂loss/∂logits (from Stage 9)

Compute:
  ∂loss/∂W_output = final_repr^T @ ∂loss/∂logits
  ∂loss/∂b_output = ∂loss/∂logits
  ∂loss/∂final_repr = ∂loss/∂logits @ W_output^T
```

**Concrete example:**

```
Correct answer: token 8 ('6')
Predicted probabilities after softmax:
  p = [0.01, 0.02, ..., 0.45, ..., 0.05]  (token 8 has p=0.45, should be 1.0)

Gradient w.r.t. logits:
  ∂loss/∂logits[8] = 0.45 - 1.0 = -0.55  (should increase)
  ∂loss/∂logits[5] = 0.15 - 0.0 = 0.15   (should decrease)
  ...

Gradient w.r.t. W_output:
  ∂loss/∂W_output[:, 8] = final_repr × (-0.55)
  
  If final_repr[0] = 0.722:
    ∂W_output[0, 8] = 0.722 × (-0.55) = -0.397
    
  Update (with learning_rate = 0.001):
    W_output[0, 8] += -0.001 × (-0.397) = W_output[0, 8] + 0.000397
```

**Effect of update:**
- Increases weights that produced correct answer
- Decreases weights that produced wrong answers
- Magnitude depends on both the error and the representation strength

---

## Deep Dive: Failure Modes and Debugging

### Common Issues in Output Projection

**Problem 1: Overconfident Wrong Predictions**

```
Symptom: Model predicts '5' with 99% confidence, but answer is '9'
Diagnosis: final_repr is in wrong region of space
Solution: Problem is in Stages 1-7, not Stage 8
```

**Problem 2: Uniform Predictions**

```
Symptom: All logits are similar (e.g., [0.1, 0.2, 0.15, ...])
Diagnosis: W_output hasn't specialized yet (early training)
Solution: Continue training, check learning rate
```

**Problem 3: Exploding Logits**

```
Symptom: Logits are huge (e.g., [45.2, -38.9, 52.3, ...])
Diagnosis: W_output weights are too large
Solution: Reduce learning rate, add weight decay, check initialization
```

**Problem 4: Predicting Syntax Tokens**

```
Symptom: Model sometimes predicts '(' or ','
Diagnosis: b_output for syntax tokens isn't negative enough
Solution: Add hard constraints or stronger bias regularization
```

**Problem 5: No Learning**

```
Symptom: Logits don't change after many epochs
Diagnosis: Gradients not flowing, possibly due to vanishing gradients
Solution: Check gradient magnitudes at each layer, verify backprop
```

---

## Connection to Real-World Transformers

Our toy transformer's output projection is **identical in principle** to production models:

### GPT-3 (175B parameters)
- **Output projection**: (12,288 → 50,257)
- **Same structure**: Linear projection from final representation to vocabulary
- **Differences**: 
  - Much larger vocabulary (50,257 vs 20)
  - Much larger embedding dimension (12,288 vs 64)
  - Uses weight tying (W_output = W_embedding^T)

### BERT (340M parameters)
- **Output projection**: (1,024 → 30,522)
- **Same structure**: Linear projection to vocabulary
- **Differences**:
  - Projects from masked positions, not just first position
  - Uses weight tying

### Our Model (102,016 parameters)
- **Output projection**: (64 → 20)
- **Same structure**: Linear projection
- **Simpler vocabulary**: 20 tokens vs 30,000-50,000

**Key insight:** Regardless of model size, the output projection is always a simple linear layer. The complexity is in learning good representations, not in the final classification!

---

## Practical Exercise: Analyzing W_output

### Exercise 1: Computing Predictions by Hand

Given:
```
final_repr = [0.5, -0.3, 0.8, -0.2, 0.6, -0.1]  (simplified to 6D)

W_output (6×5 for 5 tokens):
           [PAD]  '1'   '2'   '3'   '4'
Dim 0:     -0.5   0.3   0.4   0.5   0.6
Dim 1:      0.8  -0.2  -0.1   0.0   0.1
Dim 2:     -0.3   0.6   0.7   0.8   0.9
Dim 3:      0.4  -0.4  -0.3  -0.2  -0.1
Dim 4:     -0.2   0.5   0.6   0.7   0.8
Dim 5:      0.6  -0.3  -0.2  -0.1   0.0

b_output = [-2.0, 0.1, 0.2, 0.3, 0.4]
```

**Task:** Compute logits for all 5 tokens by hand. Which token has the highest logit?

**Solution:**
```
logits[0] = 0.5×(-0.5) + (-0.3)×0.8 + 0.8×(-0.3) + (-0.2)×0.4 + 0.6×(-0.2) + (-0.1)×0.6 + (-2.0)
          = -0.25 - 0.24 - 0.24 - 0.08 - 0.12 - 0.06 - 2.0
          = -2.99

logits[1] = 0.5×0.3 + (-0.3)×(-0.2) + 0.8×0.6 + (-0.2)×(-0.4) + 0.6×0.5 + (-0.1)×(-0.3) + 0.1
          = 0.15 + 0.06 + 0.48 + 0.08 + 0.30 + 0.03 + 0.1
          = 1.20

[Continue for logits[2], logits[3], logits[4]]
```

### Exercise 2: Gradient Computation

Given the same W_output and:
```
Correct answer: token 3 ('3')
Current probabilities: [0.01, 0.15, 0.20, 0.30, 0.34]
```

**Task:** Compute ∂loss/∂W_output[:, 3] (gradient for token '3' column)

**Solution:**
```
∂loss/∂logits[3] = probability[3] - 1 = 0.30 - 1 = -0.70

∂loss/∂W_output[:, 3] = final_repr × (-0.70)
                       = [0.5, -0.3, 0.8, -0.2, 0.6, -0.1] × (-0.70)
                       = [-0.35, 0.21, -0.56, 0.14, -0.42, 0.07]
```

This means W_output[:, 3] should be updated by adding a small multiple of final_repr to increase the logit for token 3.

---

## What's Next?

We've completed Stage 8! The representation has been transformed into logits. Next comes the final steps:

**Stage 9: Softmax & Loss**
- Convert logits to probabilities using softmax
- Compute cross-entropy loss
- Measure how far predictions are from targets
- Understand numerical stability tricks

**Stage 10: Backpropagation**
- Calculate gradients for all 102,016 parameters
- Flow backward through all stages
- Update weights to reduce loss
- Complete the training loop

**Stage 11: Training Dynamics**
- How parameters evolve over epochs
- What patterns emerge during training
- Convergence behavior and learning curves
- Attention pattern evolution

Together with Stages 9-11, we'll complete the full understanding of transformer training from input to gradient updates!

---

## Final Summary

**Stage 8 accomplishes:**

✓ Transforms abstract representations (64D) into concrete token scores (20D)

✓ Uses a simple linear projection: logits = final_repr @ W_output + b_output

✓ Each column of W_output specializes for one vocabulary token

✓ The bias term b_output encodes prior probabilities

✓ Logits are unbounded; highest value indicates the prediction

✓ The linear layer works because representations are already well-separated

✓ Only 1,300 parameters (1.3% of total model)

✓ Training gradually specializes W_output through gradient descent

✓ Mathematically identical to multi-class logistic regression

✓ Same fundamental structure as GPT, BERT, and all transformers

**The big picture:** Stage 8 is where all the sophisticated feature learning from previous stages culminates in a simple, interpretable prediction mechanism. The transformer does the hard work of creating excellent representations, and Stage 8 just needs to linearly separate them!