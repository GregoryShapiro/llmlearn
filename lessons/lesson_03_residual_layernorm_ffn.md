# Transformer Deep Dive: Stages 4-6
## Residual Connections, Layer Normalization, and Feed-Forward Networks

---

## Introduction: Stabilizing and Transforming Representations

After Stage 3's multi-head attention has allowed each token to gather contextual information from all other tokens, we need to:
1. **Stabilize** the attention output (Stage 4: Add & Norm)
2. **Transform** each position's representation independently (Stage 5: Feed-Forward Network)
3. **Stabilize** again (Stage 6: Add & Norm)

These stages might seem like mere "housekeeping" compared to attention's dramatic information gathering, but they're absolutely critical. Without them:
- Gradients would vanish or explode
- Training would be unstable or impossible
- The model couldn't learn complex transformations
- Performance would degrade catastrophically

Let's understand why each component exists and how they work together.

---

## STAGE 4: Add & Norm (After Attention)

### The Two Problems We're Solving

After multi-head attention, we have an output with shape (8 × 64). But we face two critical challenges:

**Problem 1: Gradient Flow**
In deep networks, gradients can vanish (become too small) or explode (become too large) as they backpropagate through many layers. With 2 transformer blocks, each containing attention + FFN, we have effectively 4 "layers" that gradients must flow through.

**Problem 2: Activation Instability**
The output of attention can have widely varying magnitudes:
- Some positions might have large activations: [15.2, -8.9, 23.4, ...]
- Others might have small activations: [0.01, -0.02, 0.03, ...]
- This variance makes training unstable and slow

**The Solution: Residual Connections + Layer Normalization**

---

### Part 1: Residual Connections (The "Add")

#### The Core Idea

Instead of replacing the input with the attention output, we **add** them together:

```
output = input + attention(input)
```

This creates a "shortcut" or "skip connection" that bypasses the attention layer.

#### Mathematical Definition

```
Let:
  x ∈ ℝ^(8 × 64)           = input (from Stage 2)
  F(x) ∈ ℝ^(8 × 64)        = attention output (from Stage 3)
  
Residual connection:
  y = x + F(x)
```

#### Concrete Example

For position 4 ("6") after attention:

```
Input x[4] (after positional encoding):
  [-0.87, -0.59, -0.29, 0.12, ..., 1.09]

Attention output F(x)[4]:
  [0.23, -0.12, 0.45, -0.34, ..., 0.56]

Residual output y[4]:
  y[4] = x[4] + F(x)[4]
       = [-0.87, -0.59, -0.29, 0.12, ..., 1.09]
       + [0.23, -0.12, 0.45, -0.34, ..., 0.56]
       = [-0.64, -0.71, 0.16, -0.22, ..., 1.65]
```

**Complete sequence after residual addition:**

```
Position 0 (Max):
  input: [-0.13, 0.41, -0.02, ..., 1.07]
  attn:  [0.34, -0.23, 0.56, ..., -0.12]
  sum:   [0.21, 0.18, 0.54, ..., 0.95]

Position 1 (():
  input: [0.71, 0.83, 0.15, ..., 1.11]
  attn:  [-0.12, 0.34, -0.23, ..., 0.45]
  sum:   [0.59, 1.17, -0.08, ..., 1.56]

Position 2 (1):
  input: [1.36, -0.65, 0.12, ..., 0.66]
  attn:  [0.45, -0.12, 0.23, ..., -0.34]
  sum:   [1.81, -0.77, 0.35, ..., 0.32]

... (similar for positions 3-7)
```

#### Why Residual Connections Work

**1. Gradient Highway**

During backpropagation, the gradient flows through two paths:

```
Forward:  y = x + F(x)

Backward: ∂Loss/∂x = ∂Loss/∂y × (1 + ∂F/∂x)
                    = ∂Loss/∂y + ∂Loss/∂y × ∂F/∂x
                      ↑           ↑
                   identity    through F
```

The identity term (1) ensures gradients always flow, even if ∂F/∂x is small or zero!

**Concrete gradient example:**

```
Suppose at position 4:
  ∂Loss/∂y[4] = [0.5, -0.3, 0.2, ..., 0.8]
  ∂F/∂x[4] = [0.1, 0.05, 0.02, ..., 0.08]  (small, would vanish)

Without residual:
  ∂Loss/∂x[4] = [0.5, -0.3, 0.2, ..., 0.8] × [0.1, 0.05, 0.02, ..., 0.08]
              = [0.05, -0.015, 0.004, ..., 0.064]  (much smaller!)

With residual:
  ∂Loss/∂x[4] = [0.5, -0.3, 0.2, ..., 0.8] × (1 + [0.1, 0.05, 0.02, ..., 0.08])
              = [0.5, -0.3, 0.2, ..., 0.8] × [1.1, 1.05, 1.02, ..., 1.08]
              = [0.55, -0.315, 0.204, ..., 0.864]  (gradient preserved!)
```

**2. Identity Initialization**

At the start of training, if F(x) learns to output nearly zero, the model still works:
```
y = x + F(x) ≈ x + 0 = x
```

The model starts as an identity function and gradually learns to add meaningful transformations. This makes training much more stable.

**3. Feature Reuse**

The model can choose to:
- **Preserve** important features from x (via the identity path)
- **Modify** features that need changing (via F(x))
- **Combine** both old and new information

**4. Deeper Networks**

Residual connections enable training of very deep networks. Without them:
- 10 layers: gradients vanish
- 20 layers: training fails completely

With residuals:
- 50 layers: trains well
- 100+ layers: still possible (ResNet-152, etc.)

#### Value Ranges After Residual Addition

**Before residual (attention output only):**
- Range: approximately [-5, 5]
- Mean: close to 0
- Standard deviation: ~1.5

**After residual (x + F(x)):**
- Range: approximately [-6, 6] to [-10, 10]
- Mean: close to 0
- Standard deviation: ~2-3 (larger, more variance)

The range grows because we're adding two signals together. This is actually good—it means we're accumulating information—but it's why we need normalization next.

---

### Part 2: Layer Normalization (The "Norm")

#### The Problem Residual Doesn't Solve

After adding the residual, our activations might look like:

```
Position 0: [0.21, 0.18, 0.54, ..., 0.95]     (reasonable)
Position 1: [0.59, 1.17, -0.08, ..., 1.56]    (larger)
Position 2: [1.81, -0.77, 0.35, ..., 0.32]    (one dimension huge!)
Position 3: [-8.23, 2.45, -3.12, ..., 5.67]   (very large variance!)
```

Problems:
1. **Different scales across positions**: Position 3 has much larger values
2. **Different scales across dimensions**: Some dimensions dominate others
3. **Unbounded growth**: Values could grow indefinitely through layers
4. **Unstable training**: Large values → large gradients → parameter explosions

#### What is Layer Normalization?

Layer Normalization (LayerNorm) standardizes the activations for each position independently, making them have:
- **Mean = 0**
- **Standard deviation = 1**

Then it applies learnable scaling and shifting to give the model flexibility.

#### Mathematical Definition

For each position i independently:

```
Step 1: Compute statistics across all dimensions
  μ_i = (1/d_model) × Σ_{j=0}^{d_model-1} x_{i,j}
  
  σ_i² = (1/d_model) × Σ_{j=0}^{d_model-1} (x_{i,j} - μ_i)²

Step 2: Normalize
  x̂_{i,j} = (x_{i,j} - μ_i) / √(σ_i² + ε)

Step 3: Scale and shift (learnable parameters)
  y_{i,j} = γ_j × x̂_{i,j} + β_j

Where:
  d_model = 64 (embedding dimension)
  ε = 1e-5 (small constant for numerical stability)
  γ ∈ ℝ^64 (learnable scale parameter)
  β ∈ ℝ^64 (learnable shift parameter)
```

#### Step-by-Step Example

**Position 4 (the "6") after residual:**

```
x[4] = [-0.64, -0.71, 0.16, -0.22, 0.45, 0.89, -0.34, 0.12, ..., 1.65]
       (64 values total)
```

**Step 1: Compute mean**

```
μ_4 = (1/64) × (-0.64 - 0.71 + 0.16 - 0.22 + 0.45 + 0.89 - 0.34 + 0.12 + ... + 1.65)
    = (1/64) × 15.36
    = 0.24
```

**Step 2: Compute variance**

```
For each dimension j:
  (x[4,j] - μ_4)²

Examples:
  (-0.64 - 0.24)² = (-0.88)² = 0.7744
  (-0.71 - 0.24)² = (-0.95)² = 0.9025
  (0.16 - 0.24)² = (-0.08)² = 0.0064
  ...
  (1.65 - 0.24)² = (1.41)² = 1.9881

σ_4² = (1/64) × (0.7744 + 0.9025 + 0.0064 + ... + 1.9881)
     = (1/64) × 89.28
     = 1.395

σ_4 = √1.395 = 1.181
```

**Step 3: Normalize each dimension**

```
x̂[4,0] = (-0.64 - 0.24) / 1.181 = -0.88 / 1.181 = -0.745
x̂[4,1] = (-0.71 - 0.24) / 1.181 = -0.95 / 1.181 = -0.804
x̂[4,2] = (0.16 - 0.24) / 1.181 = -0.08 / 1.181 = -0.068
x̂[4,3] = (-0.22 - 0.24) / 1.181 = -0.46 / 1.181 = -0.390
x̂[4,4] = (0.45 - 0.24) / 1.181 = 0.21 / 1.181 = 0.178
x̂[4,5] = (0.89 - 0.24) / 1.181 = 0.65 / 1.181 = 0.550
...
x̂[4,63] = (1.65 - 0.24) / 1.181 = 1.41 / 1.181 = 1.194

x̂[4] = [-0.745, -0.804, -0.068, -0.390, 0.178, 0.550, ..., 1.194]
```

**Verify normalization:**
```
Mean of x̂[4] ≈ 0.0
Std of x̂[4] ≈ 1.0
```

**Step 4: Apply learnable scale and shift**

Initially (before training):
```
γ = [1.0, 1.0, 1.0, ..., 1.0]  (64 ones)
β = [0.0, 0.0, 0.0, ..., 0.0]  (64 zeros)
```

After training, γ and β learn optimal values:
```
γ = [1.2, 0.8, 1.5, 0.9, ..., 1.1]
β = [0.1, -0.2, 0.3, -0.1, ..., 0.2]
```

Final output:
```
y[4,0] = 1.2 × (-0.745) + 0.1 = -0.794
y[4,1] = 0.8 × (-0.804) + (-0.2) = -0.843
y[4,2] = 1.5 × (-0.068) + 0.3 = 0.198
y[4,3] = 0.9 × (-0.390) + (-0.1) = -0.451
y[4,4] = 1.0 × 0.178 + 0.0 = 0.178
...

y[4] = [-0.794, -0.843, 0.198, -0.451, 0.178, ..., 1.513]
```

#### Complete Sequence After LayerNorm

```
Position 0 (Max):
  Before: [0.21, 0.18, 0.54, ..., 0.95]
  μ = 0.28, σ = 0.85
  After:  [-0.082, -0.118, 0.306, ..., 0.788]

Position 1 (():
  Before: [0.59, 1.17, -0.08, ..., 1.56]
  μ = 0.45, σ = 1.12
  After:  [0.125, 0.643, -0.473, ..., 0.991]

Position 2 (1):
  Before: [1.81, -0.77, 0.35, ..., 0.32]
  μ = 0.35, σ = 1.45
  After:  [1.007, -0.772, 0.000, ..., -0.021]

... (similar for positions 3-7)
```

#### Why LayerNorm Works

**1. Covariate Shift Reduction**

As the network trains, the distribution of inputs to each layer keeps changing. LayerNorm stabilizes these distributions, making training more consistent.

**2. Improved Gradient Flow**

Normalized activations lead to better-behaved gradients:
```
If x is normalized: x̂ ~ N(0, 1)
Then ∂Loss/∂x is also better behaved
```

**3. Learning Rate Insensitivity**

With LayerNorm, the effective learning rate becomes more consistent across different parameters and layers.

**4. Model Capacity**

The learnable γ and β parameters (128 values: 64 for scale, 64 for shift) give the model flexibility to "undo" the normalization if needed for specific dimensions.

#### Value Ranges After LayerNorm

**After normalization (before γ, β):**
- Mean: exactly 0
- Standard deviation: exactly 1
- Range: approximately [-3, 3] (99.7% of values)

**After scaling and shifting (after γ, β):**
- Mean: close to 0 (but not exactly)
- Standard deviation: close to 1 (but varies by dimension)
- Range: approximately [-3, 3] still

The key point: activations are now in a stable, predictable range.

---

### LayerNorm vs BatchNorm

You might have heard of Batch Normalization (BatchNorm). Why don't we use it?

**BatchNorm:**
```
Normalizes across the batch dimension
For position i, feature j:
  μ_j = mean across all examples in batch
  σ_j² = variance across all examples in batch
  x̂_{i,j} = (x_{i,j} - μ_j) / √(σ_j² + ε)
```

**LayerNorm:**
```
Normalizes across the feature dimension
For position i:
  μ_i = mean across all features
  σ_i² = variance across all features
  x̂_{i,j} = (x_{i,j} - μ_i) / √(σ_i² + ε)
```

**Why LayerNorm for Transformers:**

1. **Batch size independence**: LayerNorm works even with batch size = 1. BatchNorm needs large batches for good statistics.

2. **Sequence length independence**: Different sequences can have different lengths. BatchNorm would struggle.

3. **Recurrent-friendliness**: For autoregressive generation (one token at a time), LayerNorm works perfectly. BatchNorm would fail.

4. **Simplicity**: No need to track running statistics (unlike BatchNorm during inference).

**Comparison table:**

| Property | BatchNorm | LayerNorm |
|----------|-----------|-----------|
| Normalizes over | Batch dimension | Feature dimension |
| Needs batch size > 1 | Yes | No |
| Different stats at train/test | Yes | No |
| Works with variable lengths | No | Yes |
| Good for CNNs | Yes | OK |
| Good for Transformers | No | Yes |

---

### Parameter Count for Add & Norm

**Residual connection (Add):**
- Parameters: **0** (just addition, no learning)

**Layer Normalization (Norm):**
- γ: 64 parameters (one per dimension)
- β: 64 parameters (one per dimension)
- Total: **128 parameters**

This is tiny compared to attention (16,384 params) but crucial for stability!

---

### Gradient Flow Through Add & Norm

**Backward through LayerNorm:**

The gradient ∂Loss/∂x flows backward through:
1. Scale and shift (γ, β)
2. Normalization (divide by σ)
3. Mean centering (subtract μ)

**Simplified gradient (ignoring γ, β for clarity):**

```
∂Loss/∂x_{i,j} = (1/σ_i) × (∂Loss/∂x̂_{i,j} - mean_j(∂Loss/∂x̂_{i,:}))

This involves:
  - Division by σ (keeps gradient bounded)
  - Centering (removes drift)
```

The key: gradients are **normalized**, preventing them from exploding or vanishing!

**Backward through residual:**

```
Forward:  y = x + F(x)
Backward: ∂Loss/∂x = ∂Loss/∂y + ∂Loss/∂F(x)
```

The gradient splits into two paths, and the identity path ensures flow.

**Combined effect:**

Residual connections + LayerNorm create a very stable gradient highway through the network!

---

### Common Variations: Pre-Norm vs Post-Norm

Our architecture uses **Post-Norm** (norm after residual):
```
y = LayerNorm(x + Attention(x))
```

An alternative is **Pre-Norm** (norm before attention):
```
y = x + Attention(LayerNorm(x))
```

**Trade-offs:**

**Post-Norm (our choice):**
- ✓ Original Transformer paper design
- ✓ Stronger gradient signal to embeddings
- ✗ Can be less stable for very deep networks
- ✗ Requires careful initialization

**Pre-Norm:**
- ✓ More stable training (especially for deep models)
- ✓ Less sensitive to hyperparameters
- ✗ Can underfit on small datasets
- ✗ Slightly worse final performance sometimes

For our 2-layer model, Post-Norm is fine. For 50+ layers, Pre-Norm might be better.

---

## STAGE 5: Feed-Forward Network

### What's the Purpose?

After attention has gathered information from relevant positions, each position now has a **contextual representation**. But this representation is just a weighted mixture of the original embeddings—it's still somewhat "simple."

The **Feed-Forward Network (FFN)** adds non-linear transformation capacity. It allows the model to:
- Compute complex functions of the gathered information
- Create new features not present in the attention output
- Add expressiveness beyond what attention can achieve

Think of it this way:
- **Attention**: Gathers information (what's relevant?)
- **FFN**: Processes information (what do we do with it?)

### Mathematical Definition

The FFN is a **position-wise** fully connected network with one hidden layer:

```
FFN(x) = ReLU(x W₁ + b₁) W₂ + b₂

Where:
  x ∈ ℝ^(8 × 64)        Input (from Stage 4)
  W₁ ∈ ℝ^(64 × 256)     First weight matrix
  b₁ ∈ ℝ^256             First bias
  W₂ ∈ ℝ^(256 × 64)     Second weight matrix
  b₂ ∈ ℝ^64              Second bias
```

**Position-wise means:** Each of the 8 positions is processed independently using the **same** W₁, W₂, b₁, b₂. There's no interaction between positions in the FFN (that's what attention was for).

### Architecture Breakdown

**Step 1: Expansion (64 → 256)**

```
hidden = x W₁ + b₁

Input:  (8 × 64)
W₁:     (64 × 256)
Output: (8 × 256)
```

This expands from 64 dimensions to 256 dimensions (4× expansion). Why?
- Creates a richer representation space
- Allows more complex feature combinations
- Standard practice: FFN hidden dim = 4 × model dim

**Step 2: Non-linearity (ReLU)**

```
activated = ReLU(hidden) = max(0, hidden)

For each element:
  if hidden[i,j] > 0:  activated[i,j] = hidden[i,j]
  else:                activated[i,j] = 0
```

ReLU introduces non-linearity, allowing the network to learn complex functions.

**Step 3: Contraction (256 → 64)**

```
output = activated W₂ + b₂

Input:  (8 × 256)
W₂:     (256 × 64)
Output: (8 × 64)
```

This projects back to the original dimension (64), creating the final FFN output.

### Concrete Example: Position 4 (The "6")

**Input (from Stage 4, after Add & Norm):**
```
x[4] = [-0.794, -0.843, 0.198, -0.451, 0.178, ..., 1.513]  (64 dims)
```

**Step 1: First linear layer**

```
W₁ is 64 × 256, randomly initialized (truncated normal)
b₁ is 256, initialized to zeros

For dimension 0 of hidden layer:
  hidden[4,0] = Σ(x[4,j] × W₁[j,0]) + b₁[0]
              = (-0.794 × 0.023) + (-0.843 × -0.145) + ... + (1.513 × 0.089) + 0.0
              = -0.018 + 0.122 + ... + 0.135
              = 2.347

Similarly for all 256 dimensions:
  hidden[4] = [2.347, -1.234, 0.456, -3.789, 1.234, ..., 0.891]  (256 dims)
```

**Step 2: ReLU activation**

```
activated[4,0] = max(0, 2.347) = 2.347      ✓ positive, keep
activated[4,1] = max(0, -1.234) = 0.0       ✗ negative, zero out
activated[4,2] = max(0, 0.456) = 0.456      ✓ positive, keep
activated[4,3] = max(0, -3.789) = 0.0       ✗ negative, zero out
activated[4,4] = max(0, 1.234) = 1.234      ✓ positive, keep
...

activated[4] = [2.347, 0.0, 0.456, 0.0, 1.234, ..., 0.891]  (256 dims)
```

Approximately **50% of values become zero** (sparse activation). This is a key property of ReLU.

**Step 3: Second linear layer**

```
W₂ is 256 × 64, randomly initialized
b₂ is 64, initialized to zeros

For dimension 0 of output:
  output[4,0] = Σ(activated[4,k] × W₂[k,0]) + b₂[0]
              = (2.347 × 0.034) + (0.0 × -0.156) + ... + (0.891 × 0.067) + 0.0
              = 0.080 + 0.0 + ... + 0.060
              = 0.782

output[4] = [0.782, -0.456, 0.234, -0.123, 0.567, ..., -0.345]  (64 dims)
```

**Summary for position 4:**
```
Input:     64 dims  →  [-0.794, -0.843, 0.198, ..., 1.513]
Hidden:    256 dims →  [2.347, -1.234, 0.456, ..., 0.891]
Activated: 256 dims →  [2.347, 0.0, 0.456, ..., 0.891]  (~50% zeros)
Output:    64 dims  →  [0.782, -0.456, 0.234, ..., -0.345]
```

### Processing All Positions

The same W₁, W₂, b₁, b₂ are applied to **every position**:

```
Position 0 (Max):  [input] → FFN → [output]
Position 1 (():    [input] → FFN → [output]
Position 2 (1):    [input] → FFN → [output]
Position 3 (,):    [input] → FFN → [output]
Position 4 (6):    [input] → FFN → [output]
Position 5 (,):    [input] → FFN → [output]
Position 6 (2):    [input] → FFN → [output]
Position 7 ()):    [input] → FFN → [output]
```

Each gets transformed independently, but using shared weights.

**Full output shape:**
```
Input:  (8 × 64)
Output: (8 × 64)
```

The sequence length and dimension stay the same!

### Why This Architecture?

**1. Position-wise processing:**
After attention has mixed information across positions, FFN refines each position's representation independently.

**2. Expansion then contraction (bottleneck):**
```
64 → 256 → 64
```

This creates an "information bottleneck":
- Expansion: Increase capacity temporarily
- Contraction: Force compression back to original size
- Result: Learn useful compressed features

**3. Non-linearity (ReLU):**
Without ReLU, the FFN would be:
```
output = x W₁ W₂ + x W₁ b₂ + b₁ W₂ + b₁ b₂
       = x (W₁ W₂) + (bias terms)
       = x W_combined + b_combined
```

This is just a single linear transformation! Multiple linear layers without non-linearity = one linear layer.

ReLU allows learning of **non-linear** functions, which is essential for complex tasks.

**4. 4× expansion factor:**
Why 256 (4 × 64) specifically?
- Empirically found to work well
- Enough capacity without overfitting
- Computational efficiency (powers of 2)

Alternatives:
- 2× (128): Less capacity, might underfit
- 8× (512): More capacity, might overfit, slower

### What Does FFN Learn?

The FFN learns to compute complex, non-linear functions of the attention output. After training, we might observe:

**For Max operation:**
- Hidden neurons detect combinations like: "operation is Max AND first number is small"
- Output computes: "emphasize largest value in representation"

**For First operation:**
- Hidden neurons detect: "operation is First AND position is 2"
- Output computes: "emphasize first argument position"

**Feature combinations:**
The 256 hidden dimensions allow learning 256 different feature detectors, then combining them to produce 64 output features.

Example learned features:
```
Hidden[0]: "Is this a Max operation?" (fires for Max token)
Hidden[1]: "Is this the largest digit?" (fires for maximum value)
Hidden[2]: "Is this in argument position?" (fires for positions 2, 4, 6)
...

Output[0] = 0.5×Hidden[0] + 0.3×Hidden[1] + 0.2×Hidden[2] + ...
         "Combine: operation type, value magnitude, position"
```

### Parameter Count

**First layer (W₁, b₁):**
```
W₁: 64 × 256 = 16,384 parameters
b₁: 256 = 256 parameters
Total: 16,640 parameters
```

**Second layer (W₂, b₂):**
```
W₂: 256 × 64 = 16,384 parameters
b₂: 64 = 64 parameters
Total: 16,448 parameters
```

**Total FFN parameters:**
```
16,640 + 16,448 = 33,088 parameters
```

This is **twice** the parameters of attention (16,384)! FFN is the most parameter-heavy part of a transformer block.

For our full model with 2 transformer blocks:
```
2 blocks × 33,088 = 66,176 FFN parameters
```

That's ~60% of all model