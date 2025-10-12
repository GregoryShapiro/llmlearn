# Transformer from Scratch - Task List

## Phase 1: Data Pipeline ✓ (Start Here)

### Task 1.1: Vocabulary & Tokenization
- [ ] Define vocabulary dictionary (20 tokens)
- [ ] Create reverse vocabulary (index → token)
- [ ] Write `tokenize()` function: text → token indices
- [ ] Write `detokenize()` function: indices → text
- [ ] Test with sample inputs

**Files to create**: `vocabulary.py`

### Task 1.2: Data Generation
- [ ] Write `generate_example()` function
  - [ ] Random operation selection
  - [ ] Random digit generation (0-9)
  - [ ] Compute correct answer
  - [ ] Format as input sequence
- [ ] Generate 10,000 examples
- [ ] Split into train/val/test (80/10/10)
- [ ] Save to files or keep in memory

**Files to create**: `data_generator.py`

### Task 1.3: Batching & Padding
- [ ] Write `create_batch()` function
- [ ] Implement padding to max length in batch
- [ ] Create attention masks for padded positions
- [ ] Test batch shapes

**Files to create**: `data_utils.py`

---

## Phase 2: Core Components ✓

### Task 2.1: Linear Layer ✓
- [x] Implement `Linear` class
  - [x] `__init__`: Initialize weights (Xavier/He)
  - [x] `forward()`: Matrix multiplication + bias
  - [x] `backward()`: Compute gradients
- [x] Test with dummy data
- [x] Verify gradient shapes
- [x] Numerical gradient verification

**Files created**: `layers.py`

### Task 2.2: Embedding Layer ✓
- [x] Implement `Embedding` class
  - [x] `__init__`: Initialize embedding matrix
  - [x] `forward()`: Lookup embeddings
  - [x] `backward()`: Gradient accumulation
- [x] Test embedding lookup
- [x] Verify shape: (batch, seq_len) → (batch, seq_len, embed_dim)
- [x] Test gradient accumulation for repeated tokens

**Files updated**: `layers.py`

### Task 2.3: Positional Encoding ✓
- [x] Implement `PositionalEncoding` class
  - [x] `__init__`: Precompute sinusoidal encodings
  - [x] `forward()`: Add to embeddings
- [x] Verify positional encoding properties (bounded, unique)
- [x] Test shape preservation
- [x] Verify gradient flow

**Files updated**: `layers.py`

### Task 2.4: Layer Normalization ✓
- [x] Implement `LayerNorm` class
  - [x] `__init__`: Learnable scale and shift
  - [x] `forward()`: Normalize + scale + shift
  - [x] `backward()`: Compute gradients
- [x] Test normalization: mean ≈ 0, var ≈ 1
- [x] Verify gradient flow
- [x] Verify learnable parameters (gamma, beta)

**Files updated**: `layers.py`

### Task 2.5: Activation Functions ✓
- [x] Implement `ReLU` class
  - [x] `forward()`: max(0, x)
  - [x] `backward()`: gradient masking
- [x] Implement `Softmax` class
  - [x] `forward()`: exp(x) / sum(exp(x))
  - [x] `backward()`: Jacobian computation
- [x] Test numerical stability
- [x] Numerical gradient verification for Softmax

**Files updated**: `layers.py`

---

## Phase 3: Attention Mechanism ✓

### Task 3.1: Scaled Dot-Product Attention ✓
- [x] Implement `scaled_dot_product_attention()` function
  - [x] Compute Q @ K^T
  - [x] Scale by sqrt(d_k)
  - [x] Apply softmax
  - [x] Multiply by V
- [x] Test with dummy Q, K, V matrices
- [x] Verify attention weights sum to 1

**Files created**: `src/layers/attention.py`

### Task 3.2: Multi-Head Attention ✓
- [x] Implement `MultiHeadAttention` class
  - [x] `__init__`: Create Q, K, V, O projection matrices
  - [x] Split into multiple heads
  - [x] Apply attention per head
  - [x] Concatenate and project
  - [x] Implement forward pass
- [x] Test with single example
- [x] Verify output shape matches input shape

**Files updated**: `src/layers/attention.py`

### Task 3.3: Attention Backward Pass ✓
- [x] Implement gradient computation for attention
  - [x] Softmax gradient
  - [x] Q, K, V gradients
  - [x] Head splitting/concatenation gradients
- [x] Test with gradient flow verification
- [x] Verify gradient shapes

**Files updated**: `src/layers/attention.py`
**Test files created**: `tests/test_attention.py`

---

## Phase 4: Transformer Block

### Task 4.1: Feed-Forward Network
- [ ] Implement `FeedForward` class
  - [ ] Two linear layers with ReLU
  - [ ] Forward pass: Linear → ReLU → Linear
  - [ ] Backward pass
- [ ] Test dimensions: (embed_dim → ffn_dim → embed_dim)
- [ ] Verify gradient flow

**Files to create**: `transformer.py`

### Task 4.2: Transformer Block
- [ ] Implement `TransformerBlock` class
  - [ ] Multi-head attention sub-layer
  - [ ] Add & Norm (residual + layer norm)
  - [ ] Feed-forward sub-layer
  - [ ] Add & Norm again
  - [ ] Forward pass through complete block
- [ ] Test single block
- [ ] Verify residual connections work

**Files to create**: Update `transformer.py`

### Task 4.3: Complete Transformer Model
- [ ] Implement `Transformer` class
  - [ ] Embedding layer
  - [ ] Positional encoding
  - [ ] Stack of N transformer blocks
  - [ ] Output projection layer
  - [ ] Forward pass through entire model
- [ ] Test end-to-end forward pass
- [ ] Verify output shape: (batch, vocab_size)

**Files to create**: Update `transformer.py`

---

## Phase 5: Training Infrastructure

### Task 5.1: Loss Function
- [ ] Implement `CrossEntropyLoss` class
  - [ ] `forward()`: -log(p[correct_class])
  - [ ] `backward()`: Gradient of loss
- [ ] Test with dummy predictions
- [ ] Handle numerical stability (log(0))

**Files to create**: `loss.py`

### Task 5.2: Optimizer
- [ ] Implement `Adam` optimizer
  - [ ] Track momentum and velocity for each parameter
  - [ ] Update rule with bias correction
  - [ ] Apply weight decay (optional)
- [ ] Alternative: Start with simpler SGD
- [ ] Test parameter updates

**Files to create**: `optimizer.py`

### Task 5.3: Training Loop
- [ ] Implement `train_one_epoch()` function
  - [ ] Loop over batches
  - [ ] Forward pass
  - [ ] Compute loss
  - [ ] Backward pass
  - [ ] Update parameters
  - [ ] Track training metrics
- [ ] Implement `evaluate()` function
  - [ ] Validation accuracy
  - [ ] No gradient computation
- [ ] Add progress logging

**Files to create**: `train.py`

### Task 5.4: Gradient Checking (Optional but Recommended)
- [ ] Implement numerical gradient computation
- [ ] Compare with analytical gradients
- [ ] Debug any discrepancies
- [ ] Remove after verification

**Files to create**: `gradient_check.py`

---

## Phase 6: Evaluation & Visualization

### Task 6.1: Metrics & Logging
- [ ] Track training loss per epoch
- [ ] Track validation accuracy
- [ ] Save best model checkpoint
- [ ] Plot training curves
- [ ] Create confusion matrix (which operations fail?)

**Files to create**: `evaluation.py`

### Task 6.2: Attention Visualization
- [ ] Extract attention weights from model
- [ ] Plot attention heatmaps
  - [ ] Each head separately
  - [ ] Average across heads
- [ ] Analyze which heads learn what patterns
- [ ] Verify positional attention for First/Second/Last

**Files to create**: `visualization.py`

### Task 6.3: Model Testing
- [ ] Test on held-out test set
- [ ] Try hand-crafted edge cases
- [ ] Analyze failure modes
- [ ] Document what model learned

**Files to create**: `test.py`

---

## Phase 7: Polish & Documentation

### Task 7.1: Code Organization
- [ ] Organize into proper package structure
- [ ] Add docstrings to all classes/functions
- [ ] Add type hints
- [ ] Clean up debug code

### Task 7.2: README & Examples
- [ ] Write comprehensive README
- [ ] Add usage examples
- [ ] Document architecture decisions
- [ ] Include sample outputs

### Task 7.3: Experiments & Analysis
- [ ] Experiment with hyperparameters
- [ ] Try different model sizes
- [ ] Ablation studies (remove components)
- [ ] Document findings

---

## Testing Checklist (Ongoing)

### Shape Tests
- [ ] Embedding: (B, L) → (B, L, D)
- [ ] Positional: (B, L, D) → (B, L, D)
- [ ] Attention: (B, L, D) → (B, L, D)
- [ ] FFN: (B, L, D) → (B, L, D)
- [ ] Output: (B, L, D) → (B, V)

### Gradient Tests
- [ ] All parameters receive gradients
- [ ] No NaN or Inf gradients
- [ ] Gradient magnitudes reasonable
- [ ] Numerical gradient checking passes

### Training Tests
- [ ] Loss decreases over time
- [ ] Can overfit on small dataset (10 examples)
- [ ] Validation accuracy improves
- [ ] Model doesn't diverge

---

## Quick Start: First Day Tasks

**Goal: Get something running end-to-end (even if it doesn't work well)**

1. [ ] Task 1.1: Vocabulary & Tokenization
2. [ ] Task 1.2: Data Generation (100 examples)
3. [ ] Task 2.2: Embedding Layer
4. [ ] Task 2.3: Positional Encoding
5. [ ] Task 4.3: Dummy Transformer (just pass through)
6. [ ] Task 5.1: Loss Function
7. [ ] Task 5.3: Basic Training Loop

**Milestone**: Train for 1 epoch and see loss print

---

## Priority Order

**Critical Path** (must have):
1. Data pipeline
2. Embeddings + Positional encoding
3. Multi-head attention
4. Transformer block
5. Training loop

**Important** (should have):
6. Layer normalization
7. Proper optimizer (Adam)
8. Evaluation metrics

**Nice to have**:
9. Attention visualization
10. Gradient checking
11. Advanced analysis

---

## Notes

- Start simple: get something running first
- Test each component independently
- Use small examples for debugging (batch_size=1)
- Print shapes everywhere initially
- Remove print statements once working
- Commit after each working task