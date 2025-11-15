# Chapter 4: Transformer Architecture and Large Language Models - Detailed Plan

**Estimated Reading Time:** 9-10 hours
**Prerequisites:** Chapter 3, solid understanding of backpropagation, matrix operations
**Learning Objectives:**
- Understand the attention mechanism and why it revolutionized NLP
- Implement transformers from scratch using NumPy
- Master multi-head attention, positional encoding, and layer normalization
- Build and train a complete transformer model
- Understand the path from transformers to modern LLMs (GPT, BERT)
- Apply transformers to sequence tasks

**Note:** This chapter uses the existing transformer implementation in this repository (`src/transformer.py`, `src/layers/attention.py`) as reference and hands-on material.

---

## 4.1 Limitations of Traditional Neural Networks

**Duration:** 40 minutes

### Content Outline:

1. **Sequential Data Challenges** (12 min)
   - Many real-world problems involve sequences:
     - Text: Words in a sentence
     - Time series: Stock prices over time
     - Audio: Sound waves
     - Video: Frames in sequence
   - Fixed-size MLPs can't handle variable-length sequences
   - Need architecture aware of order and relationships

2. **The Recurrent Neural Network (RNN) Approach** (10 min)
   - Process sequences step-by-step
   - Hidden state carries information forward
   - Problems:
     - Sequential processing (can't parallelize)
     - Vanishing/exploding gradients over long sequences
     - Struggles with long-range dependencies
   - LSTMs/GRUs help but don't solve fundamental issues

3. **The Parallelization Problem** (8 min)
   - RNNs must process sequentially: t₁ → t₂ → t₃ → ...
   - Can't use GPUs effectively
   - Training is slow for long sequences
   - Inference is slow (matters for production)

4. **Long-Range Dependencies** (10 min)
   - Example: "The cat, which we found in the garden last summer, was hungry"
   - Subject-verb agreement: "cat...was" (not "were")
   - RNNs struggle when distance > ~20 tokens
   - Need mechanism to directly connect distant positions
   - Enter: **Attention mechanism**

### Exercise 4.1a: Analyzing Sequence Processing Challenges
**Type:** Conceptual analysis (25-30 min)

**Task:**
1. **Analyze computational complexity:**
   - MLP: O(1) per position (independent)
   - RNN: O(n) sequential steps (can't parallelize)
   - What if sequence length n = 1000?

2. **Identify long-range dependency examples:**
   - Find 5 sentences where meaning depends on words far apart
   - Example: "The keys, which I thought I lost yesterday, were in my pocket"
   - Mark the dependent words and distance

3. **Calculate RNN gradient flow:**
   - Given RNN formula: `h_t = tanh(W_h h_{t-1} + W_x x_t)`
   - Gradient from t=10 to t=0 involves 10 matrix multiplications
   - If largest eigenvalue of W_h is 0.9: (0.9)^10 ≈ 0.35 (vanishing)
   - If largest eigenvalue is 1.1: (1.1)^10 ≈ 2.59 (exploding)

4. **Reflection:**
   - Why can't we just use positional indices as features in MLP?
   - What would be ideal: Fast + long-range + parallelizable?

**Expected Outcome:**
- Clear understanding of RNN limitations
- Motivation for attention mechanism
- Appreciation for transformer innovation

### Exercise 4.1b: Understanding the Vanishing Gradient Problem
**Type:** Programming and visualization (35-45 min)

**Task:**
1. **Implement simple RNN:**
   ```python
   class SimpleRNN:
       def __init__(self, input_size, hidden_size):
           self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
           self.W_x = np.random.randn(hidden_size, input_size) * 0.01
           self.b = np.zeros((hidden_size, 1))

       def forward(self, X):
           # X shape: (input_size, seq_len)
           # TODO: Implement forward pass, track hidden states
           pass
   ```

2. **Measure gradient magnitudes:**
   - Create sequence of length 50
   - Compute gradients at each position
   - Plot: Position vs gradient magnitude
   - Try different W_h initializations

3. **Visualize vanishing:**
   - Plot gradient magnitude from output back to position 0, 10, 20, 30, 40
   - Exponential decay should be visible

4. **Compare to transformer (preview):**
   - Transformer gradients don't decay with distance
   - Direct connections via attention

**Expected Outcome:**
- Empirical observation of vanishing gradients
- Understanding why long sequences are problematic
- Setup for attention as solution

---

## 4.2 The Attention Mechanism

**Duration:** 90 minutes

### Content Outline:

1. **Core Idea: Weighted Averaging** (15 min)
   - At each position, look at ALL other positions
   - Decide how much to "attend" to each
   - Take weighted average based on relevance
   - Example: Translating "Je suis étudiant" → "I am a student"
     - When generating "student", attend to "étudiant"
     - When generating "am", attend to "suis"

2. **Query, Key, Value (QKV) Framework** (20 min)
   - **Query:** What am I looking for?
   - **Key:** What do I offer?
   - **Value:** What information do I contain?
   - Analogy: Database/search engine
     - Query = search terms
     - Keys = indexed titles
     - Values = actual content
   - Similarity score: Query · Key (dot product)
   - Higher score = more relevant

3. **Scaled Dot-Product Attention Formula** (25 min)
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) V
   ```
   - **Step 1:** Compute scores: `QK^T`
     - Shape: (seq_len_q, seq_len_k)
   - **Step 2:** Scale by `√d_k`
     - Prevents gradients from vanishing when d_k is large
     - Keeps dot products in reasonable range for softmax
   - **Step 3:** Softmax to get attention weights
     - Shape: (seq_len_q, seq_len_k)
     - Each row sums to 1 (probability distribution)
   - **Step 4:** Weighted sum of values
     - Result shape: (seq_len_q, d_v)

   Mathematical insight:
   - If Q and K are normalized, dot product measures similarity
   - Softmax converts similarities to probabilities
   - Result is expectation over values

4. **Self-Attention** (15 min)
   - Special case: Q, K, V all come from same sequence
   - Each position attends to every position (including itself)
   - Captures relationships within sequence
   - Example: "The animal didn't cross the street because it was too tired"
     - "it" should attend to "animal" (not "street")

5. **Masked Attention (Preview)** (8 min)
   - Decoder-only models (GPT): Can't look at future
   - Mask out future positions (set to -∞ before softmax)
   - Ensures causality: Position t can only attend to ≤ t
   - Not used in encoder-only (BERT) or our digit operations model

6. **Why Scaling by √d_k?** (7 min)
   - Dot product of random vectors grows with dimension
   - For d_k dimensional vectors, variance of dot product ∝ d_k
   - Softmax with large inputs: Most weight on max (too peaked)
   - Scaling keeps variance around 1
   - Preserves gradient flow

### Exercise 4.2a: Computing Attention by Hand
**Type:** Mathematical (45-55 min)

**Task:**
1. **Given small example:**
   - Sequence length: 3
   - Embedding dim: 2
   ```
   Q = [[1, 0],
        [0, 1],
        [1, 1]]

   K = [[1, 0],
        [0, 1],
        [1, 1]]

   V = [[1, 2],
        [3, 4],
        [5, 6]]

   d_k = 2
   ```

2. **Compute step-by-step:**
   - **Scores:** QK^T (show all 9 dot products)
   - **Scaled scores:** Divide by √2
   - **Attention weights:** Softmax each row
   - **Output:** Weighted sum of V

3. **Analyze results:**
   - Which positions attend to which?
   - What is the output for each query?
   - How would changing V affect output?

4. **Verify with code:**
   ```python
   def scaled_dot_product_attention(Q, K, V):
       # TODO: Implement
       pass
   ```

**Expected Outcome:**
- Deep understanding of attention mechanics
- Comfort with matrix operations in attention
- Intuition for attention weights

### Exercise 4.2b: Visualizing Attention Weights
**Type:** Programming and visualization (40-50 min)

**Task:**
1. **Create toy sequence:**
   ```python
   sentence = "The cat sat on the mat"
   # Create simple embeddings (random or pretrained)
   ```

2. **Compute self-attention:**
   ```python
   # Use your scaled_dot_product_attention function
   attention_weights, output = scaled_dot_product_attention(Q, K, V)
   ```

3. **Visualize attention matrix:**
   ```python
   import matplotlib.pyplot as plt
   plt.imshow(attention_weights, cmap='viridis')
   plt.xticks(range(len(words)), words, rotation=90)
   plt.yticks(range(len(words)), words)
   plt.xlabel('Key/Value positions')
   plt.ylabel('Query positions')
   plt.colorbar(label='Attention weight')
   plt.title('Self-Attention Pattern')
   ```

4. **Interpret:**
   - Which words attend to which?
   - Does "cat" attend to "sat"? (subject-verb)
   - Does "on" attend to "mat"? (prepositional object)

5. **Experiment:**
   - Try different sentences
   - Try different embedding dimensions
   - What patterns emerge?

**Expected Outcome:**
- Ability to visualize attention
- Understanding of attention patterns
- Foundation for multi-head attention

### Exercise 4.2c: Implementing Scaled Dot-Product Attention
**Type:** Programming (50-60 min)

**Task:**
1. **Complete implementation:**
   ```python
   def scaled_dot_product_attention(Q, K, V, mask=None):
       """
       Q: (batch, seq_len_q, d_k)
       K: (batch, seq_len_k, d_k)
       V: (batch, seq_len_v, d_v)
       mask: (batch, seq_len_q, seq_len_k) - optional

       Returns:
       - output: (batch, seq_len_q, d_v)
       - attention_weights: (batch, seq_len_q, seq_len_k)
       """
       # TODO: Implement attention formula
       # 1. Compute scores: Q @ K^T
       # 2. Scale by sqrt(d_k)
       # 3. Apply mask if provided (set masked positions to -inf)
       # 4. Softmax
       # 5. Apply attention to V
       pass
   ```

2. **Test with different inputs:**
   - Different sequence lengths (Q and K can be different!)
   - Different dimensions
   - With and without masking

3. **Implement causal mask:**
   ```python
   def create_causal_mask(seq_len):
       """
       Create mask for autoregressive attention
       Returns lower triangular matrix
       """
       # TODO
       pass
   ```

4. **Verify shapes:**
   - Input: Q (2, 10, 64), K (2, 15, 64), V (2, 15, 128)
   - Output should be: (2, 10, 128)

**Expected Outcome:**
- Working attention implementation
- Understanding of batched operations
- Masking capability for future transformer variants

### Exercise 4.2d: Understanding Query-Key-Value Concept
**Type:** Conceptual and visual (30-35 min)

**Task:**
1. **Database analogy:**
   - Imagine database of books:
     - Keys: Titles/subjects
     - Values: Book contents
     - Query: "machine learning tutorial"
   - How does attention work like search?

2. **Create visualization:**
   - Draw diagram showing Q, K, V flow
   - Show how similarity is computed
   - Show how weights are applied to values

3. **Experiment:**
   ```python
   # Create Q, K, V with specific patterns
   # Q: Looking for "position 3"
   Q = [[0, 0, 1, 0, 0]]  # One-hot at position 3

   # K: Position indicators
   K = [[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]]

   # V: Actual content
   V = [[10, 20],
        [30, 40],
        [50, 60],  # This should be selected
        [70, 80],
        [90, 100]]

   # Run attention - should output approximately [50, 60]
   ```

4. **Questions:**
   - What happens if Q matches multiple K equally?
   - What if V contains irrelevant information?
   - Why separate Q, K, V instead of just using input directly?

**Expected Outcome:**
- Intuitive understanding of QKV paradigm
- Ability to explain attention to others
- Foundation for multi-head attention

---

## 4.3 Transformer Architecture Deep Dive

**Duration:** 150 minutes (multiple subsections)

### 4.3.1 Embeddings and Positional Encoding

**Duration:** 45 minutes

#### Content Outline:

1. **Token Embeddings** (15 min)
   - Map discrete tokens to continuous vectors
   - Vocabulary: Set of all possible tokens
   - Embedding matrix: (vocab_size, embed_dim)
   - Learnable parameters (trained via backprop)
   - Similar tokens should have similar embeddings

2. **Why Positional Encoding?** (10 min)
   - Attention is permutation-invariant
   - "cat chased dog" vs "dog chased cat" would look identical!
   - Need to inject position information
   - Two approaches:
     - Learned positional embeddings (BERT)
     - Fixed sinusoidal functions (original Transformer)

3. **Sinusoidal Positional Encoding** (15 min)
   ```
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   ```
   - pos: position in sequence (0, 1, 2, ...)
   - i: dimension index (0, 1, 2, ..., d_model/2)
   - Properties:
     - Different frequency for each dimension
     - Low dimensions: Fast oscillation
     - High dimensions: Slow oscillation
     - Can extrapolate to longer sequences than seen in training
     - PE(pos+k) can be expressed as linear function of PE(pos)

4. **Combining Embeddings and Positions** (5 min)
   - Simply add: `input = token_embedding + positional_encoding`
   - Alternative: concatenate (less common, doubles dimension)
   - Addition preserves both semantic and positional info

#### Exercise 4.3.1a: Implementing Token Embeddings
**Type:** Programming (35-40 min)

**Task:**
1. **Implement Embedding layer:**
   ```python
   class Embedding:
       def __init__(self, vocab_size, embed_dim):
           # TODO: Initialize embedding matrix
           # Shape: (vocab_size, embed_dim)
           # Use small random values
           pass

       def forward(self, indices):
           # indices: (batch_size, seq_len) of integers
           # Return: (batch_size, seq_len, embed_dim)
           # TODO: Index into embedding matrix
           pass

       def backward(self, grad_output):
           # grad_output: (batch_size, seq_len, embed_dim)
           # TODO: Accumulate gradients for each token
           # Multiple positions may use same token
           pass
   ```

2. **Test:**
   - Vocab size: 20
   - Embed dim: 64
   - Input: [2, 5, 3, 7] (4 tokens)
   - Output shape should be: (4, 64)

3. **Verify gradient accumulation:**
   - If token 5 appears twice, its gradient should be sum of both
   - Implement simple check

4. **Compare to library:**
   ```python
   # If using PyTorch for verification:
   import torch.nn as nn
   emb = nn.Embedding(20, 64)
   ```

**Expected Outcome:**
- Working embedding layer
- Understanding of lookup table operation
- Gradient accumulation for repeated tokens

#### Exercise 4.3.1b: Sinusoidal Positional Encoding from Scratch
**Type:** Programming (50-60 min)

**Task:**
1. **Implement positional encoding:**
   ```python
   def create_positional_encoding(max_seq_len, embed_dim):
       """
       Create sinusoidal positional encoding matrix

       Returns:
       - pe: (max_seq_len, embed_dim)
       """
       pe = np.zeros((max_seq_len, embed_dim))

       # TODO: Implement formula
       # For each position:
       #   For each dimension:
       #     If even: sin(pos / 10000^(i/embed_dim))
       #     If odd: cos(pos / 10000^((i-1)/embed_dim))

       return pe
   ```

2. **Verify properties:**
   - Each position has unique encoding
   - Encoding for position p+k is linear combination of position p
   - Test: Can you compute PE(5) from PE(3) and PE(4)?

3. **Visualize:**
   - Heatmap: Position vs Dimension
   - Line plot: A few dimensions over positions
   - Show different frequencies for different dimensions

4. **Compare to reference:**
   - Check against `src/layers/positional_encoding.py` in this repo

**Expected Outcome:**
- Working positional encoding
- Visual understanding of sinusoidal patterns
- Appreciation for elegant mathematical solution

#### Exercise 4.3.1c: Visualizing Position Embeddings
**Type:** Visualization and analysis (30-35 min)

**Task:**
1. **Create position embeddings:**
   - Max length: 100
   - Embedding dim: 128
   - Generate PE matrix

2. **Visualize patterns:**
   ```python
   # Heatmap
   plt.figure(figsize=(12, 8))
   plt.imshow(pe.T, aspect='auto', cmap='RdBu')
   plt.xlabel('Position')
   plt.ylabel('Embedding Dimension')
   plt.title('Sinusoidal Positional Encoding')
   plt.colorbar()
   ```

3. **Analyze specific dimensions:**
   - Plot dimensions 0, 10, 50, 100 over positions
   - Show frequency differences

4. **Similarity between positions:**
   - Compute cosine similarity between all pairs
   - Plot similarity matrix
   - Nearby positions should be more similar

5. **Extrapolation test:**
   - Train on sequences up to length 50
   - Test on sequences length 75
   - Does it work? (Should, with sinusoidal)

**Expected Outcome:**
- Deep visual understanding of positional encoding
- Appreciation for generalization properties
- Ready to use in transformer

---

### 4.3.2 Multi-Head Attention

**Duration:** 50 minutes

#### Content Outline:

1. **Motivation for Multiple Heads** (10 min)
   - Single attention: One perspective on relationships
   - Multi-head: Multiple perspectives simultaneously
   - Example:
     - Head 1: Syntactic relationships (subject-verb)
     - Head 2: Semantic relationships (synonyms)
     - Head 3: Positional relationships (adjacent words)
   - Different heads learn different patterns

2. **Architecture** (15 min)
   - Parameters:
     - `num_heads`: Number of parallel attention mechanisms (commonly 8)
     - `d_model`: Total model dimension (e.g., 512)
     - `d_k = d_v = d_model / num_heads` (e.g., 64 per head)

   - Linear projections (learned):
     - W_Q: (d_model, d_model)
     - W_K: (d_model, d_model)
     - W_V: (d_model, d_model)
     - W_O: (d_model, d_model) - output projection

   - Process:
     1. Project input to Q, K, V
     2. Split into num_heads chunks
     3. Apply attention in parallel for each head
     4. Concatenate head outputs
     5. Apply output projection

3. **Computation Steps** (15 min)
   ```python
   # Input: X (batch, seq_len, d_model)

   # 1. Project to Q, K, V
   Q = X @ W_Q  # (batch, seq_len, d_model)
   K = X @ W_K
   V = X @ W_V

   # 2. Split into heads
   Q = reshape(Q, (batch, seq_len, num_heads, d_k))
   Q = transpose(Q, (batch, num_heads, seq_len, d_k))
   # Same for K, V

   # 3. Scaled dot-product attention per head
   attention_output = scaled_dot_product_attention(Q, K, V)
   # (batch, num_heads, seq_len, d_v)

   # 4. Concatenate heads
   concat = transpose(attention_output, (batch, seq_len, num_heads, d_v))
   concat = reshape(concat, (batch, seq_len, d_model))

   # 5. Output projection
   output = concat @ W_O
   ```

4. **Computational Complexity** (10 min)
   - Attention: O(n²d) where n is sequence length, d is dimension
   - Memory: O(n²) for attention weights matrix
   - Parallelization: All heads computed in parallel
   - Bottleneck for very long sequences (>2048)

#### Exercise 4.3.2a: Single-Head vs Multi-Head Comparison
**Type:** Experimental (35-45 min)

**Task:**
1. **Implement single-head attention:**
   ```python
   class SingleHeadAttention:
       def __init__(self, d_model):
           # Single set of Q, K, V projections
           pass
   ```

2. **Implement multi-head attention:**
   ```python
   class MultiHeadAttention:
       def __init__(self, d_model, num_heads):
           assert d_model % num_heads == 0
           self.d_k = d_model // num_heads
           # TODO: Initialize projections
           pass
   ```

3. **Train on simple task:**
   - Sequence copying: [1, 2, 3, 4] → [1, 2, 3, 4]
   - Single head vs 4 heads

4. **Compare:**
   - Convergence speed
   - Final accuracy
   - Attention patterns (visualize)

**Expected Outcome:**
- Understanding of multi-head benefits
- Working multi-head implementation
- Empirical validation

#### Exercise 4.3.2b: Implementing Multi-Head Attention
**Type:** Programming (70-85 min)

**Task:**
1. **Complete implementation with backward pass:**
   ```python
   class MultiHeadAttention:
       def __init__(self, embed_dim, num_heads):
           self.embed_dim = embed_dim
           self.num_heads = num_heads
           self.head_dim = embed_dim // num_heads

           # Initialize projections (use Xavier initialization)
           self.W_Q = ...
           self.W_K = ...
           self.W_V = ...
           self.W_O = ...

       def forward(self, X, mask=None):
           batch_size, seq_len, _ = X.shape

           # TODO: Implement multi-head attention
           # 1. Linear projections
           # 2. Split into heads
           # 3. Scaled dot-product attention (per head)
           # 4. Concatenate
           # 5. Output projection

           # Cache for backward pass
           self.cache = {...}

           return output, attention_weights

       def backward(self, grad_output):
           # TODO: Implement backward pass
           # Reverse of forward operations
           pass
   ```

2. **Test shape correctness:**
   - Input: (2, 10, 64)
   - 8 heads
   - Output: (2, 10, 64)
   - Attention weights: (2, 8, 10, 10)

3. **Gradient checking:**
   - Use numerical gradients to verify
   - Check all parameters: W_Q, W_K, W_V, W_O

4. **Compare to reference:**
   - Check against `src/layers/attention.py` in this repo

**Expected Outcome:**
- Production-ready multi-head attention
- Complete forward/backward implementation
- Verified with gradient checking

#### Exercise 4.3.2c: Analyzing Different Attention Heads
**Type:** Visualization and analysis (40-50 min)

**Task:**
1. **Train model with 4 heads on simple task:**
   - Digit operations (from this repo's task)
   - "Max ( 3 5 1 )" → 5

2. **Extract attention patterns from each head:**
   ```python
   # After training, run forward pass
   _, attention_weights = multi_head_attn.forward(X)
   # attention_weights: (batch, num_heads, seq_len, seq_len)

   # Visualize each head separately
   for head in range(num_heads):
       plot_attention_heatmap(attention_weights[0, head])
   ```

3. **Analyze patterns:**
   - Does head 1 focus on operation token?
   - Does head 2 focus on numbers?
   - Does head 3 focus on syntax ( , ) ?
   - Does head 4 do something else?

4. **Compute attention entropy:**
   ```python
   # High entropy = distributed attention
   # Low entropy = focused attention
   entropy = -sum(p * log(p))
   ```

5. **Cluster heads:**
   - Group heads with similar patterns
   - Are some heads redundant?

**Expected Outcome:**
- Understanding that different heads learn different patterns
- Visualization skills for attention
- Insight into what transformer learns

---

### 4.3.3 Feed-Forward Networks

**Duration:** 25 minutes

#### Content Outline:

1. **Architecture** (8 min)
   - Two linear layers with activation in between
   ```
   FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   ```
   - Dimensions:
     - Input/Output: d_model (e.g., 512)
     - Hidden: 4 * d_model (e.g., 2048)
   - ReLU activation (or GeLU in modern variants)

2. **Position-wise Application** (7 min)
   - Applied independently to each position
   - Same FFN used for all positions
   - Can be viewed as 1x1 convolution
   - Adds non-linearity and expressiveness

3. **Why Expand then Project?** (10 min)
   - Expansion creates higher-dimensional space
   - More capacity for complex transformations
   - Projection back to d_model for residual connection
   - 4x expansion is empirical sweet spot
   - Most parameters in transformer are in FFN layers

#### Exercise 4.3.3a: Implementing Position-wise FFN
**Type:** Programming (30-35 min)

**Task:**
1. **Implement FFN:**
   ```python
   class FeedForward:
       def __init__(self, d_model, d_ff):
           # d_ff typically 4 * d_model
           # TODO: Initialize W1, b1, W2, b2
           pass

       def forward(self, X):
           # X: (batch, seq_len, d_model)
           # TODO:
           # 1. Linear: X @ W1 + b1
           # 2. ReLU
           # 3. Linear: @ W2 + b2
           # Cache for backward
           pass

       def backward(self, grad_output):
           # TODO: Backprop through both layers
           pass
   ```

2. **Verify shapes:**
   - Input: (2, 10, 512)
   - Hidden: (2, 10, 2048) after first linear
   - Output: (2, 10, 512)

3. **Test position-wise property:**
   - FFN(X[:, i, :]) should equal FFN(X)[:, i, :]
   - Verify this holds

4. **Gradient check:**
   - Use numerical gradients
   - Verify both W1 and W2 gradients

**Expected Outcome:**
- Working FFN implementation
- Understanding of position-wise operation
- Ready to integrate into transformer block

#### Exercise 4.3.3b: Understanding Dimension Expansion
**Type:** Experimental (30-40 min)

**Task:**
1. **Try different expansion ratios:**
   - 1x (no expansion): d_ff = d_model
   - 2x: d_ff = 2 * d_model
   - 4x: d_ff = 4 * d_model
   - 8x: d_ff = 8 * d_model

2. **Train simple model with each:**
   - Small transformer with 2 layers
   - Digit operations task

3. **Compare:**
   - Parameter count
   - Training time
   - Final accuracy
   - Overfitting behavior

4. **Analysis:**
   - Is 4x optimal for this task?
   - Does 8x overfit?
   - Does 1x underfit?

**Expected Outcome:**
- Understanding of capacity vs overfitting trade-off
- Empirical validation of 4x rule
- Model design intuition

---

### 4.3.4 Layer Normalization and Residual Connections

**Duration:** 40 minutes

#### Content Outline:

1. **The Gradient Flow Problem** (8 min)
   - Deep networks suffer from vanishing/exploding gradients
   - Even worse than RNNs because of transformer depth
   - Need mechanism to preserve gradient flow

2. **Residual Connections** (12 min)
   - Original ResNet idea (2015)
   - Instead of `y = F(x)`, use `y = F(x) + x`
   - Creates "gradient highway"
   - Gradient splits: ∂L/∂x = ∂L/∂y * (∂F/∂x + 1)
   - The "+1" ensures gradient flow even if ∂F/∂x ≈ 0
   - Pattern in transformer:
     ```python
     x = x + MultiHeadAttention(x)
     x = x + FeedForward(x)
     ```

3. **Layer Normalization** (15 min)
   - Normalize across features for each sample
   - Formula:
     ```
     LayerNorm(x) = γ * (x - μ) / σ + β
     ```
     - μ: mean across features
     - σ: std across features
     - γ, β: learnable parameters (scale and shift)

   - Why not Batch Norm?
     - Batch norm: Normalize across batch dimension
     - Problematic for variable-length sequences
     - Layer norm: Normalize each sample independently

   - Benefits:
     - Stabilizes training
     - Allows higher learning rates
     - Reduces internal covariate shift

4. **Pre-LN vs Post-LN** (5 min)
   - **Post-LN** (original transformer):
     ```python
     x = LayerNorm(x + MultiHeadAttention(x))
     x = LayerNorm(x + FeedForward(x))
     ```
   - **Pre-LN** (modern, more stable):
     ```python
     x = x + MultiHeadAttention(LayerNorm(x))
     x = x + FeedForward(LayerNorm(x))
     ```
   - Pre-LN easier to train deep models (>12 layers)
   - This repo uses Post-LN (following original paper)

#### Exercise 4.3.4a: Implementing Layer Normalization
**Type:** Programming (40-50 min)

**Task:**
1. **Implement LayerNorm:**
   ```python
   class LayerNorm:
       def __init__(self, normalized_shape, eps=1e-5):
           # normalized_shape: Dimensions to normalize over
           self.eps = eps
           self.gamma = np.ones(normalized_shape)
           self.beta = np.zeros(normalized_shape)

       def forward(self, X):
           # X: (batch, seq_len, d_model)
           # Normalize over last dimension (d_model)
           # TODO:
           # 1. Compute mean and variance
           # 2. Normalize
           # 3. Scale and shift with gamma, beta
           # 4. Cache for backward
           pass

       def backward(self, grad_output):
           # TODO: Derive and implement
           # This is tricky! Need:
           # - grad w.r.t. normalized input
           # - grad w.r.t. gamma
           # - grad w.r.t. beta
           pass
   ```

2. **Test:**
   - Input: (2, 10, 64)
   - Verify output mean ≈ 0, std ≈ 1 (before gamma/beta)
   - Verify gamma and beta are applied correctly

3. **Gradient checking:**
   - Numerical vs analytical gradients
   - This is complex, so thorough checking is important

4. **Compare to reference:**
   - Check against `src/layers/normalization.py` in this repo

**Expected Outcome:**
- Working layer normalization
- Understanding of normalization statistics
- Correct gradient implementation

#### Exercise 4.3.4b: Gradient Flow with Residual Connections
**Type:** Programming and analysis (35-45 min)

**Task:**
1. **Compare gradient flow:**
   - Network WITHOUT residuals:
     ```python
     x = Layer1(x)
     x = Layer2(x)
     x = Layer3(x)
     ```
   - Network WITH residuals:
     ```python
     x = x + Layer1(x)
     x = x + Layer2(x)
     x = x + Layer3(x)
     ```

2. **Measure gradients:**
   - Create 10-layer network
   - Compute gradient magnitude at each layer
   - Plot: Layer depth vs gradient magnitude

3. **With residuals:**
   - Gradient should be relatively constant

4. **Without residuals:**
   - Gradient should decay exponentially

5. **Verify mathematically:**
   - With residuals: ∂L/∂x₀ includes direct path
   - Without: Must go through all layer Jacobians

**Expected Outcome:**
- Empirical verification of residual benefits
- Understanding of gradient flow
- Appreciation for ResNet innovation

#### Exercise 4.3.4c: Pre-LN vs Post-LN Comparison
**Type:** Experimental (40-50 min)

**Task:**
1. **Implement both variants:**
   - Post-LN (original transformer):
     ```python
     x = LayerNorm(x + Attention(x))
     x = LayerNorm(x + FFN(x))
     ```
   - Pre-LN:
     ```python
     x = x + Attention(LayerNorm(x))
     x = x + FFN(LayerNorm(x))
     ```

2. **Train identical models:**
   - Same architecture, data, hyperparameters
   - Only difference: Pre-LN vs Post-LN

3. **Compare:**
   - Training stability (loss variance)
   - Convergence speed
   - Final accuracy
   - Gradient magnitudes during training

4. **Try with different depths:**
   - 2 layers (should be similar)
   - 6 layers (Pre-LN should be better)
   - 12 layers (Pre-LN should be much better)

**Expected Outcome:**
- Understanding of normalization placement
- Empirical validation of Pre-LN for deep models
- Design choice awareness

---

## 4.4 Training Transformers

**Duration:** 60 minutes

### Content Outline:

1. **Cross-Entropy Loss for Classification** (15 min)
   - For multi-class: Softmax + Cross-Entropy
   ```
   L = -Σ y_true * log(y_pred)
   ```
   - Combined with softmax, gradient simplifies:
   ```
   ∂L/∂z = y_pred - y_true
   ```
   - This beautiful simplification is why we use this combination

2. **Training Loop Structure** (12 min)
   ```python
   for epoch in range(num_epochs):
       for batch in dataloader:
           # Forward pass
           outputs = model(batch.inputs)
           loss = criterion(outputs, batch.targets)

           # Backward pass
           gradients = model.backward(loss_grad)

           # Update
           optimizer.step(gradients)

       # Validation
       val_accuracy = evaluate(model, val_data)
   ```

3. **Learning Rate Warmup** (13 min)
   - Problem: Large random gradients at start
   - Solution: Gradually increase LR for first few epochs
   - Formula:
     ```
     if step < warmup_steps:
         lr = max_lr * (step / warmup_steps)
     else:
         lr = max_lr
     ```
   - Prevents early training instability
   - Original transformer paper used this

4. **Gradient Clipping** (10 min)
   - Problem: Occasional very large gradients
   - Solution: Clip gradient norm
     ```python
     grad_norm = sqrt(sum(g**2 for all g))
     if grad_norm > max_norm:
         for g in gradients:
             g *= max_norm / grad_norm
     ```
   - Typical max_norm: 1.0 or 5.0
   - Prevents training divergence

5. **Label Smoothing (Advanced)** (10 min)
   - Instead of hard targets [0, 0, 1, 0]
   - Use soft targets [0.025, 0.025, 0.9, 0.025]
   - Prevents overconfidence
   - Improves generalization
   - Formula: `y_smooth = (1 - ε) * y_true + ε / num_classes`

### Exercise 4.4a: Implementing Cross-Entropy Loss
**Type:** Programming (35-40 min)

**Task:**
1. **Implement stable cross-entropy:**
   ```python
   class CrossEntropyLoss:
       def __init__(self):
           pass

       def forward(self, logits, targets):
           # logits: (batch, num_classes) - before softmax
           # targets: (batch,) - class indices

           # TODO:
           # 1. Compute softmax (numerically stable)
           # 2. Compute cross-entropy
           # 3. Return loss and cache for backward
           pass

       def backward(self):
           # Return gradient w.r.t. logits
           # Should be: softmax_probs - one_hot(targets)
           pass
   ```

2. **Numerical stability:**
   - Subtract max before exp (log-sum-exp trick)
   ```python
   logits_max = np.max(logits, axis=-1, keepdims=True)
   exp_logits = np.exp(logits - logits_max)
   ```

3. **Test:**
   - Verify gradient: Should be `probs - targets`
   - Test with extreme values (large positive/negative)
   - Gradient check

4. **Compare to reference:**
   - Check against `src/loss.py` in this repo

**Expected Outcome:**
- Numerically stable loss implementation
- Understanding of softmax-cross-entropy combination
- Ready for training

### Exercise 4.4b: Complete Training Loop
**Type:** Programming (60-75 min)

**Task:**
1. **Implement training infrastructure:**
   ```python
   def train_epoch(model, dataloader, optimizer, criterion):
       total_loss = 0
       for batch_idx, (inputs, targets) in enumerate(dataloader):
           # Forward
           outputs, cache = model.forward(inputs)
           loss, grad_output = criterion.forward(outputs, targets)

           # Backward
           gradients = model.backward(grad_output, cache)

           # Update
           optimizer.step(gradients)

           total_loss += loss

       return total_loss / len(dataloader)

   def evaluate(model, dataloader):
       correct = 0
       total = 0
       for inputs, targets in dataloader:
           outputs = model.forward(inputs)[0]
           predictions = np.argmax(outputs, axis=-1)
           correct += (predictions == targets).sum()
           total += len(targets)
       return correct / total
   ```

2. **Main training loop:**
   ```python
   for epoch in range(num_epochs):
       train_loss = train_epoch(model, train_loader, optimizer, criterion)
       val_accuracy = evaluate(model, val_loader)

       print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_accuracy:.4f}")

       # Early stopping check
       if val_accuracy > best_accuracy:
           best_accuracy = val_accuracy
           save_checkpoint(model, optimizer, epoch)
   ```

3. **Test on digit operations:**
   - Use dataset from this repo
   - Train for 20 epochs
   - Should achieve >90% accuracy

**Expected Outcome:**
- Complete training pipeline
- Model that actually learns
- Foundation for all experiments

### Exercise 4.4c: Gradient Clipping Implementation
**Type:** Programming (25-30 min)

**Task:**
1. **Implement gradient clipping:**
   ```python
   def clip_gradients(gradients, max_norm):
       # Compute global norm
       total_norm = 0
       for grad in gradients.values():
           total_norm += np.sum(grad ** 2)
       total_norm = np.sqrt(total_norm)

       # Clip if necessary
       if total_norm > max_norm:
           clip_coef = max_norm / (total_norm + 1e-6)
           for param_name in gradients:
               gradients[param_name] *= clip_coef

       return gradients, total_norm
   ```

2. **Integrate into training:**
   - Clip after backward, before optimizer step
   - Track gradient norms over time

3. **Experiment:**
   - Train without clipping
   - Train with clipping (max_norm=1.0)
   - Compare stability

4. **Visualize:**
   - Plot gradient norms over training
   - Show when clipping activates

**Expected Outcome:**
- Working gradient clipping
- Understanding of when it helps
- More stable training

### Exercise 4.4d: Learning Rate Warmup
**Type:** Programming (30-35 min)

**Task:**
1. **Implement warmup schedule:**
   ```python
   class WarmupSchedule:
       def __init__(self, warmup_steps, d_model):
           self.warmup_steps = warmup_steps
           self.d_model = d_model

       def get_lr(self, step):
           # Original transformer schedule:
           # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
           arg1 = step ** (-0.5)
           arg2 = step * (self.warmup_steps ** (-1.5))
           return (self.d_model ** (-0.5)) * min(arg1, arg2)
   ```

2. **Modify training to use schedule:**
   ```python
   schedule = WarmupSchedule(warmup_steps=4000, d_model=512)

   for step in range(total_steps):
       lr = schedule.get_lr(step)
       optimizer.learning_rate = lr
       # ... training step
   ```

3. **Visualize schedule:**
   - Plot LR vs step for first 10000 steps
   - Show warmup then decay

4. **Compare:**
   - No warmup (constant LR)
   - With warmup
   - Effect on early training stability

**Expected Outcome:**
- Working LR schedule
- Understanding of warmup benefits
- Improved training stability

---

## 4.5 From Transformers to Large Language Models

**Duration:** 60 minutes

### Content Outline:

1. **Three Transformer Variants** (15 min)
   - **Encoder-only** (BERT):
     - Bidirectional context
     - Masked language modeling
     - Best for: Classification, named entity recognition

   - **Decoder-only** (GPT):
     - Autoregressive (left-to-right)
     - Causal masking
     - Best for: Text generation

   - **Encoder-Decoder** (T5, BART):
     - Encoder processes input
     - Decoder generates output
     - Best for: Translation, summarization

2. **GPT Architecture** (15 min)
   - Stack of decoder blocks
   - Causal self-attention (can't see future)
   - Trained on next-token prediction
   - Scale: GPT-3 has 175B parameters, 96 layers
   - Few-shot learning emerges from scale

3. **BERT Architecture** (12 min)
   - Stack of encoder blocks
   - Bidirectional attention
   - Trained on:
     - Masked language modeling (predict masked words)
     - Next sentence prediction
   - Used for: Feature extraction, fine-tuning

4. **Our Model: Encoder-only for Classification** (10 min)
   - Similar to BERT but simpler
   - No masking during training
   - Task: Sequence → single output
   - Example: "Max ( 3 5 1 )" → 5
   - First token pooling for classification

5. **Scaling Laws** (8 min)
   - Larger models → better performance (to a point)
   - Compute, data, parameters scale together
   - Emergent abilities at scale:
     - Few-shot learning
     - Reasoning
     - Code generation
   - Chinchilla paper: Optimal compute budget

### Exercise 4.5a: Comparing Encoder vs Decoder Architectures
**Type:** Conceptual and programming (40-50 min)

**Task:**
1. **Implement causal mask for decoder:**
   ```python
   def create_causal_mask(seq_len):
       # Upper triangular matrix of -inf
       mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
       return mask
   ```

2. **Modify attention to support masking:**
   - Your attention should already support mask parameter
   - Test with causal mask

3. **Compare on task:**
   - Encoder (bidirectional): Can see full sequence
   - Decoder (causal): Can only see past

   - Task: Sequence completion
     - Input: [1, 2, 3, 4]
     - Target: [2, 3, 4, 5]

   - Which architecture is more appropriate?

4. **Analysis:**
   - When would you use encoder?
   - When would you use decoder?
   - What about encoder-decoder?

**Expected Outcome:**
- Understanding of architectural variants
- Ability to implement each type
- Task-appropriate architecture selection

### Exercise 4.5b: Masked Language Modeling
**Type:** Programming (50-60 min)

**Task:**
1. **Implement masking strategy:**
   ```python
   def create_mlm_batch(tokens, mask_prob=0.15, vocab_size=20):
       """
       Randomly mask tokens for MLM training
       Returns: masked_tokens, targets, mask_positions
       """
       masked = tokens.copy()
       targets = tokens.copy()

       # Randomly select positions to mask
       mask_positions = np.random.rand(*tokens.shape) < mask_prob

       # For each masked position:
       # 80% replace with [MASK]
       # 10% replace with random token
       # 10% keep original
       # TODO: Implement

       return masked, targets, mask_positions
   ```

2. **Train with MLM objective:**
   - Encoder-only transformer
   - Predict masked tokens
   - Loss only on masked positions

3. **Compare to supervised:**
   - MLM (self-supervised)
   - Regular supervised (with labels)
   - Which learns better representations?

4. **Analysis:**
   - How much data is "used" per example?
   - MLM: Only 15% of tokens contribute to loss
   - Trade-off: More flexible but less efficient

**Expected Outcome:**
- Understanding of self-supervised learning
- MLM implementation
- Appreciation for BERT training

### Exercise 4.5c: Autoregressive Generation
**Type:** Programming (45-55 min)

**Task:**
1. **Implement generation loop:**
   ```python
   def generate(model, start_tokens, max_length, temperature=1.0):
       """
       Autoregressively generate sequence
       """
       generated = start_tokens.copy()

       for i in range(max_length):
           # Forward pass (with causal masking)
           logits = model.forward(generated)

           # Get logits for next position
           next_logits = logits[:, -1, :] / temperature

           # Sample (or argmax for greedy)
           next_token = np.argmax(next_logits, axis=-1)

           # Append to sequence
           generated = np.concatenate([generated, next_token[:, None]], axis=1)

           # Stop if EOS token
           if next_token == EOS_TOKEN:
               break

       return generated
   ```

2. **Test generation:**
   - Start with "Max ("
   - Generate rest of sequence
   - Does it produce valid operations?

3. **Experiment with temperature:**
   - T=0.1: Nearly deterministic (argmax)
   - T=1.0: Standard sampling
   - T=2.0: More random/creative
   - Show examples for each

4. **Analysis:**
   - Quality vs diversity trade-off
   - When is each temperature appropriate?

**Expected Outcome:**
- Working autoregressive generation
- Understanding of temperature sampling
- Foundation for LLM inference

---

## 4.6 Hands-on: Implementing a Transformer from Scratch

**Duration:** 120 minutes (capstone)

### Content Outline:

This is the capstone exercise for Chapter 4, bringing together all components into a complete transformer.

### Exercise 4.6a: Building Complete Transformer (Digit Operations)
**Type:** Capstone project (150-180 min)

**Task:**
1. **Use existing implementation as reference:**
   - Review `src/transformer.py` in this repo
   - Review `src/layers/attention.py`
   - Understanding the architecture

2. **Complete any missing pieces in your implementation:**
   ```python
   class TransformerBlock:
       def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
           self.attention = MultiHeadAttention(d_model, num_heads)
           self.norm1 = LayerNorm(d_model)
           self.ffn = FeedForward(d_model, d_ff)
           self.norm2 = LayerNorm(d_model)

       def forward(self, x):
           # Post-LN architecture
           # Attention sub-layer
           attn_output = self.attention(x)
           x = self.norm1(x + attn_output)

           # FFN sub-layer
           ffn_output = self.ffn(x)
           x = self.norm2(x + ffn_output)

           return x

   class Transformer:
       def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
           self.embedding = Embedding(vocab_size, d_model)
           self.pos_encoding = create_positional_encoding(max_seq_len, d_model)
           self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
           self.output_projection = Linear(d_model, vocab_size)

       def forward(self, x):
           # Embedding + positional encoding
           x = self.embedding(x) + self.pos_encoding[:x.shape[1]]

           # Transformer blocks
           for block in self.blocks:
               x = block(x)

           # Pool first token
           x = x[:, 0, :]

           # Project to vocab
           logits = self.output_projection(x)

           return logits
   ```

3. **Load digit operations dataset:**
   ```python
   from src.data_generatpr import generate_tokenized_dataset, split_dataset

   dataset = generate_tokenized_dataset(num_examples=10000, num_args=3, max_value=9)
   train_data, val_data, test_data = split_dataset(dataset)
   ```

4. **Configure model:**
   - vocab_size: 20 (from `src/vocabluary.py`)
   - d_model: 64
   - num_heads: 4
   - num_layers: 2
   - d_ff: 256
   - max_seq_len: 50

5. **Train:**
   - Adam optimizer, lr=0.001
   - Batch size: 64
   - Epochs: 30-50
   - Target: >95% accuracy

**Expected Outcome:**
- Complete working transformer
- 95%+ accuracy on test set
- Understanding of all components

### Exercise 4.6b: Training and Evaluation
**Type:** Experimental (60-75 min)

**Task:**
1. **Comprehensive training:**
   - Track all metrics:
     - Train loss, val loss
     - Train accuracy, val accuracy
     - Per-operation accuracy
     - Gradient norms
     - Learning rate

2. **Create evaluation suite:**
   ```python
   def evaluate_per_operation(model, test_data):
       results = {op: {'correct': 0, 'total': 0} for op in operations}

       for inputs, targets, operation in test_data:
           predictions = model.predict(inputs)
           correct = (predictions == targets).sum()
           results[operation]['correct'] += correct
           results[operation]['total'] += len(targets)

       for op in results:
           acc = results[op]['correct'] / results[op]['total']
           print(f"{op}: {acc*100:.2f}%")
   ```

3. **Analysis:**
   - Which operations are easiest? (First, Second, Last)
   - Which are hardest? (Max, Min)
   - Why the difference?

4. **Compare to reference:**
   - This repo has pre-trained checkpoints
   - Compare your results
   - Should be similar (93-99%)

**Expected Outcome:**
- Thorough evaluation methodology
- Understanding of model strengths/weaknesses
- Matching reference performance

### Exercise 4.6c: Attention Visualization Analysis
**Type:** Visualization and interpretation (50-60 min)

**Task:**
1. **Extract attention weights:**
   ```python
   # Modify forward pass to return attention weights
   def forward_with_attention(self, x):
       attentions = []
       # ... (collect from each layer/head)
       return logits, attentions
   ```

2. **Visualize for specific examples:**
   ```python
   # Example: "Max ( 5 3 9 )"
   input_tokens = [15, 17, 7, 5, 11, 18]  # [Max, (, 5, 3, 9, )]

   # Run forward pass
   logits, attentions = model.forward_with_attention(input_tokens)

   # Plot attention for each layer and head
   for layer in range(num_layers):
       for head in range(num_heads):
           plot_attention_heatmap(
               attentions[layer][head],
               tokens=["Max", "(", "5", "3", "9", ")"]
           )
   ```

3. **Analyze patterns:**
   - For "First": Does it attend to position 2?
   - For "Max": Does it attend to all numbers?
   - For "Last": Does it attend to the last number position?

4. **Compare operations:**
   - Visualize all 5 operations
   - Are attention patterns operation-specific?
   - Do different heads specialize?

5. **Use visualization tools from repo:**
   - Check `src/visualization.py`
   - Use `extract_attention_weights()` and `plot_attention_heatmaps()`

**Expected Outcome:**
- Beautiful attention visualizations
- Understanding of what model learned
- Insight into transformer internals

### Exercise 4.6d: Debugging Common Transformer Issues
**Type:** Debugging practice (40-50 min)

**Task:**
1. **Common issues checklist:**

   **Issue: Loss is NaN**
   - Check: Learning rate too high?
   - Check: Gradient explosion? (use gradient clipping)
   - Check: Numerical stability in softmax/loss?

   **Issue: Loss not decreasing**
   - Check: Learning rate too low?
   - Check: Vanishing gradients? (residual connections working?)
   - Check: Wrong labels or data preprocessing?

   **Issue: High train acc, low val acc**
   - Overfitting
   - Solution: More data, regularization, simpler model

   **Issue: Slow training**
   - Check: Batch size too small?
   - Check: Unnecessary computation in forward pass?
   - Profile code to find bottleneck

2. **Deliberately introduce bugs:**
   - Remove residual connections → gradient vanishing
   - Remove layer norm → training instability
   - Wrong attention mask → information leakage
   - For each, observe and fix

3. **Create debugging guide:**
   - Symptom → likely cause → solution
   - Based on your experience

**Expected Outcome:**
- Practical debugging skills
- Understanding of component importance
- Confidence in troubleshooting

---

## Math Appendices (4A-4E)

*(Detailed mathematical treatments as outlined in the main chapter outline)*

---

## Chapter 4 Summary

**Key Takeaways:**
1. Attention enables direct connections between all positions
2. Multi-head attention learns multiple relationship types
3. Positional encoding injects position information
4. Layer norm + residuals enable deep networks
5. Transformers are the foundation of modern LLMs
6. Architecture variants (encoder/decoder) suit different tasks

**Prerequisites for Chapter 5:**
- Working transformer implementation
- Understanding of attention mechanism
- Comfort with complex architectures
- Experience with real training and debugging

**Connection to Repository:**
This chapter directly uses the existing implementation in this repo:
- `src/transformer.py` - Complete transformer
- `src/layers/attention.py` - Multi-head attention
- `src/layers/positional_encoding.py` - Sinusoidal encoding
- `src/layers/normalization.py` - Layer normalization
- `src/data_generatpr.py` - Digit operations dataset
- `train_step_by_step.py` - Training script
- `test_model_manually.py` - Interactive testing

**Total Exercises:** 19 main exercises across all subsections
**Total Time:** 9-10 hours reading + 18-22 hours exercises = **27-32 hours**

---

## Consistency Check (Internal)

**Terminology:**
- "Attention weights" = "attention scores after softmax" ✓
- "Self-attention" vs "cross-attention" distinguished ✓
- "Encoder" vs "Decoder" clearly explained ✓
- Post-LN used (matching repo implementation) ✓

**Prerequisites:**
- Assumes backpropagation mastery from Chapter 3 ✓
- Matrix operations (shape manipulations critical) ✓
- Residual connections introduced conceptually in Chapter 3 ✓

**Flow to Chapter 5:**
- Transformer is foundation for advanced architectures ✓
- CNNs and RNNs provide contrast ✓
- Vision Transformers extend attention to images ✓
- Modern trends build on transformer base ✓

**Alignment with Repository:**
- Uses same architecture (encoder-only, 2 layers, 4 heads, 64 dim) ✓
- Uses same dataset (digit operations) ✓
- Uses same vocabulary (20 tokens) ✓
- References actual file paths ✓
- Expected accuracy matches repo checkpoints (93-99%) ✓
