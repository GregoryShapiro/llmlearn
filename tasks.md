# Transformer from Scratch - Task List

## Phase 1: Data Pipeline ✓

### Task 1.1: Vocabulary & Tokenization ✓
- [x] Define vocabulary dictionary (20 tokens)
- [x] Create reverse vocabulary (index → token)
- [x] Write `tokenize()` function: text → token indices
- [x] Write `detokenize()` function: indices → text
- [x] Test with sample inputs

**Files created**: `src/vocabluary.py`

### Task 1.2: Data Generation ✓
- [x] Write `generate_example()` function
  - [x] Random operation selection
  - [x] Random digit generation (0-9)
  - [x] Compute correct answer
  - [x] Format as input sequence
- [x] Generate 10,000 examples support
- [x] Split into train/val/test (80/10/10)
- [x] Balance operations option

**Files created**: `src/data_generatpr.py`

### Task 1.3: Batching & Padding ✓
- [x] Write `create_batch()` function
- [x] Implement padding to max length in batch
- [x] Create attention masks for padded positions
- [x] Test batch shapes

**Files created**: `src/data_utils.py`

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

## Phase 4: Transformer Block ✓

### Task 4.1: Feed-Forward Network ✓
- [x] Implement `FeedForward` class
  - [x] Two linear layers with ReLU
  - [x] Forward pass: Linear → ReLU → Linear
  - [x] Backward pass
- [x] Test dimensions: (embed_dim → ffn_dim → embed_dim)
- [x] Verify gradient flow

**Files created**: `src/transformer.py`

### Task 4.2: Transformer Block ✓
- [x] Implement `TransformerBlock` class
  - [x] Multi-head attention sub-layer
  - [x] Add & Norm (residual + layer norm)
  - [x] Feed-forward sub-layer
  - [x] Add & Norm again
  - [x] Forward pass through complete block
- [x] Test single block
- [x] Verify residual connections work

**Files updated**: `src/transformer.py`

### Task 4.3: Complete Transformer Model ✓
- [x] Implement `Transformer` class
  - [x] Embedding layer
  - [x] Positional encoding
  - [x] Stack of N transformer blocks
  - [x] Output projection layer
  - [x] Forward pass through entire model
- [x] Test end-to-end forward pass
- [x] Verify output shape: (batch, vocab_size)

**Files updated**: `src/transformer.py`
**Test files created**: `tests/test_transformer.py`

---

## Phase 5: Training Infrastructure ✓

### Task 5.1: Loss Function ✓
- [x] Implement `CrossEntropyLoss` class
  - [x] `forward()`: -log(p[correct_class])
  - [x] `backward()`: Gradient of loss
- [x] Test with dummy predictions
- [x] Handle numerical stability (log(0))

**Files created**: `src/loss.py`

### Task 5.2: Optimizer ✓
- [x] Implement `Adam` optimizer
  - [x] Track momentum and velocity for each parameter
  - [x] Update rule with bias correction
  - [x] Tested adaptive learning rates
- [x] Also implemented simpler SGD with momentum
- [x] Test parameter updates

**Files created**: `src/optimizer.py`

### Task 5.3: Training Loop ✓
- [x] Implement `train_one_epoch()` function
  - [x] Loop over batches
  - [x] Forward pass
  - [x] Compute loss
  - [x] Backward pass
  - [x] Update parameters
  - [x] Track training metrics
- [x] Implement `evaluate()` function
  - [x] Validation accuracy
  - [x] No gradient computation
- [x] Add progress logging

**Files created**: `src/train_utils.py`
**Test files created**: `tests/test_training.py`

### Task 5.4: Gradient Checking (Optional but Recommended)
- [ ] Implement numerical gradient computation
- [ ] Compare with analytical gradients
- [ ] Debug any discrepancies
- [ ] Remove after verification

**Files to create**: `gradient_check.py` (Skipped for now - gradient tests in layer tests are sufficient)

---

## Phase 6: Evaluation & Visualization ✓

### Task 6.1: Metrics & Logging ✓
- [x] Track training loss per epoch
- [x] Track validation accuracy
- [x] Save best model checkpoint
- [x] Plot training curves
- [x] Create per-operation accuracy analysis

**Files created**: `src/evaluation.py`

### Task 6.2: Attention Visualization ✓
- [x] Extract attention weights framework (needs model modification)
- [x] Plot attention heatmaps (text and matplotlib versions)
  - [x] Each head separately
  - [x] All heads comparison view
  - [x] Attention evolution across layers
- [x] Analyze attention patterns automatically
- [x] Pattern detection (focused vs distributed)

**Files created**: `src/visualization.py`

### Task 6.3: Model Testing
- [ ] Test on held-out test set
- [ ] Try hand-crafted edge cases
- [ ] Analyze failure modes
- [ ] Document what model learned

**Note**: Task 6.3 will be done after training a model

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