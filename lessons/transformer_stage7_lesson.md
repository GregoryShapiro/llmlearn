# Transformer Deep Dive: Stage 7
## Transformer Block 2 - Hierarchical Feature Learning

---

## Introduction: Building on Foundations

After our input has passed through Transformer Block 1 (Stages 3-6), each token now has a rich, contextual representation. Block 1 has:
- Gathered initial contextual information via attention
- Transformed representations through the feed-forward network
- Stabilized outputs with residual connections and layer normalization

But we're not done yet. **Transformer Block 2** takes these intermediate representations and processes them further, learning higher-level patterns and more abstract relationships.

Think of it like reading comprehension:
- **Block 1:** Understanding individual words and their immediate context
- **Block 2:** Understanding sentences, relationships, and deeper meaning

For our task `Max(1,6,2)`, this means:
- **Block 1:** "This is a Max operation with three numbers: 1, 6, and 2"
- **Block 2:** "Among these three numbers, 6 is the maximum, so that's the answer"

---

## The Core Concept: Why Stack Transformer Blocks?

### The Power of Composition

A single transformer block can learn basic patterns, but complex reasoning requires **composing** multiple patterns together. Stacking blocks enables hierarchical learning:

**Mathematical Perspective:**
```
Block 1 output: f₁(x)
Block 2 output: f₂(f₁(x))

This is function composition, creating:
  f₂ ∘ f₁(x)
```

Each block can learn different aspects of the problem, and later blocks build on earlier blocks' understanding.

**Information Flow:**
```
Raw Input → Block 1 → Intermediate Features → Block 2 → Abstract Features
```

### What Makes Block 2 Different from Block 1?

**Structurally:** Identical architecture (Attention → Add & Norm → FFN → Add & Norm)

**Functionally:** Different learned weights, so different patterns

**Key differences:**

| Aspect | Block 1 | Block 2 |
|--------|---------|---------|
| **Input** | Embeddings + Positional Encoding | Block 1's processed output |
| **Attention learns** | Basic patterns (tokens, syntax) | Complex patterns (semantics, logic) |
| **FFN learns** | Simple features | Composed features |
| **Output** | Intermediate representation | Final representation for prediction |

---

## Stage 7 Architecture Overview

### The Complete Structure

Block 2 has **exactly the same architecture** as Block 1:

```
Input from Block 1 (8 × 64)
    ↓
┌──────────────────────────────────────┐
│  Multi-Head Attention                │
│  - 4 heads (different W_q, W_k, W_v) │
│  - Each head: 64 → 16 dims           │
│  - Attention computation             │
│  - Output projection (different W_o) │
│  Parameters: 16,384                  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Add & Norm                          │
│  - Residual connection               │
│  - Layer Normalization               │
│  - Different γ₃, β₃                  │
│  Parameters: 128                     │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Feed-Forward Network                │
│  - Linear: 64 → 256 (different W₃)  │
│  - ReLU activation                   │
│  - Linear: 256 → 64 (different W₄)  │
│  Parameters: 33,088                  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Add & Norm                          │
│  - Residual connection               │
│  - Layer Normalization               │
│  - Different γ₄, β₄                  │
│  Parameters: 128                     │
└──────────────────────────────────────┘
    ↓
Final Output (8 × 64)

Total Block 2 parameters: 49,728
```

### Parameter Independence

**Critical point:** Block 2 has **completely separate parameters** from Block 1.

**Block 1 parameters:**
- Attention: W_q⁽¹⁾, W_k⁽¹⁾, W_v⁽¹⁾, W_o⁽¹⁾
- LayerNorm 1: γ₁, β₁
- FFN: W₁, W₂, b₁, b₂
- LayerNorm 2: γ₂, β₂

**Block 2 parameters:**
- Attention: W_q⁽²⁾, W_k⁽²⁾, W_v⁽²⁾, W_o⁽²⁾ (different!)
- LayerNorm 3: γ₃, β₃ (different!)
- FFN: W₃, W₄, b₃, b₄ (different!)
- LayerNorm 4: γ₄, β₄ (different!)

**Total model parameters:**
```
Embeddings:           20 × 64     = 1,280
Positional Encoding:  (fixed)     = 0
Block 1:                          = 49,728
Block 2:                          = 49,728
Output Layer:         64 × 20     = 1,280
─────────────────────────────────────────
Total:                            = 102,016
```

---

## PART 1: Input to Block 2

### What Does Block 1 Output Look Like?

After passing through Block 1, our sequence `Max(1,6,2)` has been transformed:

**Block 1 Input (after embeddings + PE):**
```
Position 0 (Max): [-0.87, -0.59, -0.29, 0.12, ..., 1.09]  (64 dims)
Position 1 (():   [0.71, 0.83, 0.15, -0.23, ..., 1.11]
Position 2 (1):   [1.36, -0.65, 0.12, 0.34, ..., 0.66]
Position 3 (,):   [0.48, 0.35, -0.29, 0.67, ..., 0.12]
Position 4 (6):   [-0.87, -0.59, -0.29, 0.12, ..., 1.09]
Position 5 (,):   [0.48, 0.35, -0.29, 0.67, ..., 0.12]
Position 6 (2):   [1.05, -0.31, 0.09, -0.45, ..., 0.34]
Position 7 ()):   [0.56, 0.82, -0.11, 0.29, ..., 0.78]
```

**Block 1 Output (input to Block 2):**
```
Position 0 (Max): [-0.116, -1.279, 0.448, -0.707, ..., 0.829]  (64 dims)
Position 1 (():   [0.234, 0.567, -0.892, 0.123, ..., -0.456]
Position 2 (1):   [0.789, -0.234, 0.567, -0.123, ..., 0.345]
Position 3 (,):   [-0.345, 0.678, -0.234, 0.567, ..., -0.123]
Position 4 (6):   [-0.116, -1.279, 0.448, -0.707, ..., 0.829]
Position 5 (,):   [-0.345, 0.678, -0.234, 0.567, ..., -0.123]
Position 6 (2):   [0.567, -0.892, 0.234, -0.456, ..., 0.789]
Position 7 ()):   [-0.234, 0.789, -0.345, 0.456, ..., -0.567]
```

**Key observations:**

1. **Different from input:** Values have been transformed significantly
2. **Normalized range:** Thanks to LayerNorm, values are in [-3, 3]
3. **Contextual:** Each position now contains information from other positions
4. **Same shape:** Still (8 × 64), ready for Block 2

### What Information Does Block 1 Encode?

By the end of Block 1, the representation has learned:

**Positional Understanding:**
- Position 0 knows it's at the start (operation position)
- Positions 2, 4, 6 know they're arguments
- Positions 1, 3, 5, 7 know they're syntax tokens

**Syntactic Understanding:**
- Parentheses are matched: ( at position 1, ) at position 7
- Commas separate arguments at positions 3 and 5
- Structure is: operation(arg₁, arg₂, arg₃)

**Semantic Understanding (partial):**
- "Max" token has gathered information about all three numbers
- Each number token has some awareness of the operation type
- But deeper reasoning isn't complete yet

**What's Missing:**
- Final decision: which number is actually the maximum?
- Confidence: how certain is the model?
- Robustness: handling edge cases and variations

This is where Block 2 comes in!

---

## PART 2: Multi-Head Attention in Block 2

### The Same Mechanism, Different Learning

Block 2's attention works **identically** to Block 1's:
1. Create Q, K, V via learned projections
2. Compute attention scores
3. Apply softmax
4. Weighted sum of values
5. Output projection

But the **learned weights are different**, so the **patterns learned are different**.

### Step-by-Step: Attention in Block 2

**Input: Block 1's output**
```
X⁽²⁾ ∈ ℝ⁽⁸ ˣ ⁶⁴⁾

Position 4 (the "6"):
X⁽²⁾[4] = [-0.116, -1.279, 0.448, -0.707, ..., 0.829]
```

**Step 1: Create Q, K, V (with Block 2's weights)**

```
Q⁽²⁾ = X⁽²⁾ @ W_q⁽²⁾    where W_q⁽²⁾ ∈ ℝ⁽⁶⁴ ˣ ¹⁶⁾ (for one head)
K⁽²⁾ = X⁽²⁾ @ W_k⁽²⁾    where W_k⁽²⁾ ∈ ℝ⁽⁶⁴ ˣ ¹⁶⁾
V⁽²⁾ = X⁽²⁾ @ W_v⁽²⁾    where W_v⁽²⁾ ∈ ℝ⁽⁶⁴ ˣ ¹⁶⁾

Result: Each is (8 × 16) for one head
```

**Concrete example for Head 1, Position 4:**

```
X⁽²⁾[4] = [-0.116, -1.279, 0.448, ..., 0.829]  (64 dims)

Q⁽²⁾[4] = X⁽²⁾[4] @ W_q⁽²⁾
        = [0.67, -0.34, 0.89, -0.23, ..., 0.45]  (16 dims)

K⁽²⁾[4] = X⁽²⁾[4] @ W_k⁽²⁾
        = [0.34, -0.67, 0.23, -0.89, ..., 0.12]  (16 dims)

V⁽²⁾[4] = X⁽²⁾[4] @ W_v⁽²⁾
        = [0.89, -0.45, 0.12, -0.67, ..., 0.78]  (16 dims)
```

**Step 2: Compute Attention Scores**

```
Scores⁽²⁾ = Q⁽²⁾ @ K⁽²⁾ᵀ / √16

For position 4 attending to position 2:
scores⁽²⁾[4,2] = (Q⁽²⁾[4] · K⁽²⁾[2]) / 4
               = (0.67×0.45 + (-0.34)×(-0.23) + ... + 0.45×0.89) / 4
               = 2.34 / 4
               = 0.585
```

**Full attention scores matrix (after scaling):**

```
         0     1     2     3     4     5     6     7
      (Max)  (  )  (1)  (,)  (6)  (,)  (2)  ()
  0 [ 0.72  0.15  0.45  0.08  0.68  0.12  0.38  0.22 ]  Max
  1 [ 0.18  0.65  0.22  0.48  0.25  0.52  0.28  0.71 ]  (
  2 [ 0.55  0.12  0.78  0.18  0.62  0.15  0.48  0.19 ]  1
  3 [ 0.09  0.42  0.16  0.58  0.14  0.55  0.18  0.45 ]  ,
  4 [ 0.68  0.18  0.58  0.12  0.82  0.15  0.32  0.21 ]  6
  5 [ 0.11  0.48  0.19  0.62  0.16  0.65  0.22  0.51 ]  ,
  6 [ 0.38  0.15  0.48  0.11  0.35  0.14  0.71  0.18 ]  2
  7 [ 0.22  0.68  0.16  0.52  0.19  0.55  0.21  0.75 ]  )
```

**Step 3: Apply Softmax**

```
For position 4 (the "6"):
scores⁽²⁾[4] = [0.68, 0.18, 0.58, 0.12, 0.82, 0.15, 0.32, 0.21]

Softmax:
attention⁽²⁾[4] = [0.195, 0.119, 0.178, 0.112, 0.227, 0.116, 0.138, 0.122]
```

**Full attention weights matrix:**

```
         0     1     2     3     4     5     6     7    Sum
      (Max)  (  )  (1)  (,)  (6)  (,)  (2)  ()
  0 [ 0.198 0.108 0.156 0.099 0.193 0.104 0.141 0.111 ] 1.00  Max
  1 [ 0.103 0.179 0.109 0.145 0.113 0.150 0.115 0.186 ] 1.00  (
  2 [ 0.168 0.104 0.211 0.108 0.180 0.106 0.150 0.110 ] 1.00  1
  3 [ 0.096 0.138 0.106 0.167 0.105 0.160 0.108 0.142 ] 1.00  ,
  4 [ 0.195 0.119 0.178 0.112 0.227 0.116 0.138 0.122 ] 1.00  6
  5 [ 0.099 0.145 0.109 0.175 0.107 0.181 0.111 0.148 ] 1.00  ,
  6 [ 0.141 0.106 0.150 0.103 0.139 0.105 0.195 0.108 ] 1.00  2
  7 [ 0.111 0.193 0.106 0.150 0.109 0.160 0.111 0.210 ] 1.00  )
```

**Comparing to Block 1 Attention:**

Let's compare attention patterns for position 4 ("6"):

```
Block 1 attention[4]: [0.142, 0.114, 0.120, 0.103, 0.179, 0.111, 0.126, 0.105]
Block 2 attention[4]: [0.195, 0.119, 0.178, 0.112, 0.227, 0.116, 0.138, 0.122]
                        ↑                    ↑                     ↑
                     Higher              Higher              Higher
```

**Key difference:** Block 2 attends MORE strongly to:
- Position 0 (Max operation): 0.195 vs 0.142
- Position 2 (the "1"): 0.178 vs 0.120
- Position 6 (the "2"): 0.138 vs 0.126

**Interpretation:** Block 2 has learned to:
1. Pay MORE attention to the operation (Max)
2. Pay MORE attention to the other arguments (1 and 2)
3. Compare values more explicitly

This makes sense! Block 2 is doing the **reasoning** that Block 1 prepared for.

**Step 4: Weighted Sum of Values**

```
Output⁽²⁾[4] = Σⱼ attention⁽²⁾[4,j] × V⁽²⁾[j]

           = 0.195×V⁽²⁾[0] + 0.119×V⁽²⁾[1] + 0.178×V⁽²⁾[2] + ...
           
For dimension 0:
           = 0.195×0.78 + 0.119×(-0.34) + 0.178×0.45 + ...
           = 0.152 + (-0.040) + 0.080 + ...
           = 0.589

Complete output:
Output⁽²⁾[4] = [0.589, -0.234, 0.456, -0.789, ..., 0.912]  (16 dims for one head)
```

**Step 5: Multi-Head and Output Projection**

Same as Block 1:
- 4 heads process in parallel
- Concatenate: [Head 1 | Head 2 | Head 3 | Head 4] = 64 dims
- Project through W_o⁽²⁾: (64 × 64)

```
Final attention output from Block 2:
attention_out⁽²⁾[4] = [0.345, -0.678, 0.234, -0.456, ..., 0.789]  (64 dims)
```

### What Has Block 2's Attention Learned?

After training, Block 2's attention heads specialize differently than Block 1:

**Block 1 Heads:**
- Head 1: Operation recognition
- Head 2: Argument gathering
- Head 3: Value awareness
- Head 4: Syntax matching

**Block 2 Heads:**
- Head 1: **Operation-to-winner connection** (Max → 6)
- Head 2: **Comparative reasoning** (6 > 1 and 6 > 2)
- Head 3: **Confidence assessment** (how certain?)
- Head 4: **Error checking** (verify structure is correct)

**Visualization: Block 2 Head 1 for "Max(1,6,2)"**

```
Attention pattern showing Max → 6 connection:

         Max  (   1   ,   6   ,   2   )
    Max [ ░░  ░   ░   ░   ██  ░   ░   ░  ]  ← Strong to winning number
    (   [ ░   ░░  ░   ░   ░   ░   ░   ░░ ]
    1   [ ░   ░   ░░  ░   ░   ░   ░   ░  ]
    ,   [ ░   ░   ░   ░░  ░   ░   ░   ░  ]
    6   [ ██  ░   ░   ░   ░░  ░   ░   ░  ]  ← Looks back to operation
    ,   [ ░   ░   ░   ░░  ░   ░   ░   ░  ]
    2   [ ░   ░   ░   ░   ░   ░   ░░  ░  ]
    )   [ ░   ░░  ░   ░   ░   ░   ░   ░░ ]

Legend: ██ = high (0.7-0.9), ░ = low (0.05-0.15), ░░ = medium (0.15-0.3)
```

**Comparison: Block 1 vs Block 2 for position 4 ("6")**

```
Block 1 Head 2 (argument gathering):
  Attends to: [Max: 0.22, 1: 0.20, 6: 0.21, 2: 0.19]
  Interpretation: "These are all the arguments"

Block 2 Head 1 (winner selection):
  Attends to: [Max: 0.65, 1: 0.08, 6: 0.15, 2: 0.07]
  Interpretation: "In a Max operation, I'm the answer"
```

### The Hierarchical Pattern Learning

**Stage-by-stage refinement:**

```
Raw input:          "Max" "(" "1" "," "6" "," "2" ")"
                     ↓
After embeddings:   Token meanings + positions
                     ↓
After Block 1:      "This is Max with three numbers"
                     ↓
After Block 2:      "6 is the maximum of {1, 6, 2}"
                     ↓
After output layer: "Answer: 6" (high confidence)
```

---

## PART 3: Add & Norm After Attention (Block 2)

### Residual Connection

Same mechanism as Block 1, but operating on Block 2's data:

```
residual_out = X⁽²⁾ + attention_out⁽²⁾

Where:
  X⁽²⁾ = input to Block 2 (Block 1's output)
  attention_out⁽²⁾ = Block 2's attention output
```

**For position 4:**

```
X⁽²⁾[4] = [-0.116, -1.279, 0.448, -0.707, ..., 0.829]

attention_out⁽²⁾[4] = [0.345, -0.678, 0.234, -0.456, ..., 0.789]

residual_out[4] = [-0.116, -1.279, 0.448, -0.707, ..., 0.829]
                + [0.345, -0.678, 0.234, -0.456, ..., 0.789]
                = [0.229, -1.957, 0.682, -1.163, ..., 1.618]
```

**Why critical in Block 2:**

Block 2 is further from the input, so gradient flow is even more important:

```
Loss → Block 2 output → Block 2 attention → Block 1 output → Block 1 attention → Input

With residuals:
Loss → Block 2 output ─┬→ Block 2 attention → Block 1 output → ...
                       └────────────────────────────────────→ (shortcut!)
```

### Layer Normalization (with γ₃, β₃)

```
For each position i:
  μᵢ = mean of residual_out[i] over 64 dimensions
  σᵢ = std of residual_out[i] over 64 dimensions
  
  normalized[i,j] = (residual_out[i,j] - μᵢ) / (σᵢ + ε)
  
  final[i,j] = γ₃[j] × normalized[i,j] + β₃[j]
```

**For position 4:**

```
residual_out[4] = [0.229, -1.957, 0.682, -1.163, ..., 1.618]

μ₄ = mean([0.229, -1.957, 0.682, ...]) = -0.156
σ₄ = std([0.229, -1.957, 0.682, ...]) = 1.234

Normalize:
normalized[4,0] = (0.229 - (-0.156)) / 1.234 = 0.312
normalized[4,1] = (-1.957 - (-0.156)) / 1.234 = -1.460
normalized[4,2] = (0.682 - (-0.156)) / 1.234 = 0.679
...

Apply learned γ₃ and β₃:
γ₃ = [1.1, 0.9, 1.2, 1.0, ..., 0.8]  (learned for Block 2)
β₃ = [0.05, -0.1, 0.15, -0.05, ..., 0.1]

norm_out[4,0] = 1.1 × 0.312 + 0.05 = 0.393
norm_out[4,1] = 0.9 × (-1.460) + (-0.1) = -1.414
norm_out[4,2] = 1.2 × 0.679 + 0.15 = 0.965
...

norm_out[4] = [0.393, -1.414, 0.965, -1.050, ..., 0.645]
```

**Important:** γ₃ and β₃ are **different parameters** from Block 1's γ₁, γ₂!

**All LayerNorm parameters in the model:**
- γ₁, β₁: After Block 1's attention (64 + 64 = 128 params)
- γ₂, β₂: After Block 1's FFN (64 + 64 = 128 params)
- γ₃, β₃: After Block 2's attention (64 + 64 = 128 params)
- γ₄, β₄: After Block 2's FFN (64 + 64 = 128 params)

**Total LayerNorm parameters: 512**

---

## PART 4: Feed-Forward Network in Block 2

### Same Architecture, Different Weights

Block 2's FFN has the same structure as Block 1:
```
FFN⁽²⁾(x) = ReLU(x W₃ + b₃) W₄ + b₄

Where:
  W₃ ∈ ℝ⁽⁶⁴ ˣ ²⁵⁶⁾  (different from Block 1's W₁)
  b₃ ∈ ℝ²⁵⁶         (different from Block 1's b₁)
  W₄ ∈ ℝ⁽²⁵⁶ ˣ ⁶⁴⁾  (different from Block 1's W₂)
  b₄ ∈ ℝ⁶⁴          (different from Block 1's b₂)
```

### Step-by-Step for Position 4

**Input (from Add & Norm):**
```
x[4] = [0.393, -1.414, 0.965, -1.050, ..., 0.645]  (64 dims)
```

**Step 1: First linear layer (expansion 64 → 256)**

```
hidden[4] = x[4] @ W₃ + b₃

For dimension 0:
hidden[4,0] = Σⱼ (x[4,j] × W₃[j,0]) + b₃[0]
            = (0.393 × 0.045) + (-1.414 × -0.123) + ... + 0.0
            = 0.018 + 0.174 + ...
            = 3.456

Complete hidden layer:
hidden[4] = [3.456, -2.123, 0.789, -4.567, 1.234, ..., 2.345]  (256 dims)
```

**Step 2: ReLU activation**

```
activated[4] = max(0, hidden[4])
             = [3.456, 0.0, 0.789, 0.0, 1.234, ..., 2.345]
                      ↑           ↑
                    zeroed     zeroed
```

Approximately **52% of values** are zeroed out (slight variation from 50% due to learned biases).

**Step 3: Second linear layer (contraction 256 → 64)**

```
output[4] = activated[4] @ W₄ + b₄

For dimension 0:
output[4,0] = Σₖ (activated[4,k] × W₄[k,0]) + b₄[0]
            = (3.456 × 0.034) + (0.0 × -0.156) + (0.789 × 0.067) + ...
            = 0.118 + 0.0 + 0.053 + ...
            = 0.892

Complete output:
output[4] = [0.892, -0.567, 0.345, -0.234, ..., 0.678]  (64 dims)
```

### What Has Block 2's FFN Learned?

**Block 1's FFN learned:**
- Simple feature detectors
- "Is this a Max operation?"
- "Is this a digit?"
- "Is this in an argument position?"

**Block 2's FFN learns:**
- Complex decision features
- "Is this digit the maximum?"
- "What's the confidence level?"
- "Does this match the operation type?"

**Example learned features in hidden layer:**

```
Hidden neuron 45 (Block 1): Fires for any digit token
Hidden neuron 45 (Block 2): Fires specifically for the MAXIMUM digit

Hidden neuron 127 (Block 1): Fires for operation tokens
Hidden neuron 127 (Block 2): Fires when operation MATCHES the selected answer
```

**Visualization: Feature activation**

For input "Max(1,6,2)":

```
Block 1 FFN Hidden Layer (simplified):
Neuron 10: [0.0, 0.0, 2.3, 0.0, 2.1, 0.0, 2.5, 