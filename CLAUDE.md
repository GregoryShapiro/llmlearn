# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **from-scratch transformer implementation** using only NumPy (no PyTorch/TensorFlow). The project learns simple digit operations (First, Second, Last, Max, Min) as a toy problem to understand LLM architecture.

**Key characteristic:** This is an educational codebase prioritizing readability and understanding over optimization. Every component has extensive documentation explaining both the "what" and the "why".

## Learning Resources

**📚 For learning transformer architecture:** Start with the [lessons/](lessons/) directory, which contains a comprehensive 9-lesson series covering every stage of the transformer pipeline:

1. `lesson_01_embeddings_and_positional_encoding.md` - Stages 1-2
2. `lesson_02_attention_mechanism.md` - Stage 3 (multi-head attention)
3. `lesson_03_residual_layernorm_ffn.md` - Stages 4-6
4. `lesson_04_transformer_block_2.md` - Stage 7 (stacking blocks)
5. `lesson_05_output_projection.md` - Stage 8
6. `lesson_06_softmax_and_loss.md` - Stage 9
7. `lesson_07_backpropagation.md` - Stage 10 (gradients)
8. `lesson_08_training_loop.md` - Stage 11 (optimizers)
9. `lesson_09_training_dynamics.md` - Stage 12 (how model learns)

See [lessons/README.md](lessons/README.md) for the complete learning path (~15 hours of material).

## Common Commands

### Testing
```bash
# Run all unit tests for specific components
python3 tests/test_layers.py         # Core layers (embedding, linear, layer norm, etc.)
python3 tests/test_attention.py      # Attention mechanism
python3 tests/test_transformer.py    # Full transformer model
python3 tests/test_training.py       # Training infrastructure
python3 tests/test_integration.py    # End-to-end integration test

# No pytest configuration - tests use unittest framework
# Each test file is directly executable
```

### Training
```bash
# Interactive training with step-by-step visualization
python3 train_step_by_step.py --size small --epochs 10     # Quick test (~2-3 min)
python3 train_step_by_step.py --size medium --epochs 20    # Production (~6 min)
python3 train_step_by_step.py --size large --epochs 50 --no-interactive  # Best accuracy (2-3 hrs)

# Custom parameters
python3 train_step_by_step.py --size medium --epochs 30 --batch-size 64 --lr 0.001
```

### Manual Testing
```bash
# Test pre-trained models interactively
python3 test_model_manually.py

# Pre-trained checkpoints in checkpoints/ directory
# - Run 1: 93.5% accuracy (seed=42)
# - Run 2: 99.2% accuracy (seed=123) - best
```

## Architecture Overview

### The Transformer Flow
```
Input Tokens (e.g., "Max ( 5 3 9 )")
    ↓
Embedding Layer (20 → 64 dims)
    ↓
Positional Encoding (add position information)
    ↓
Transformer Block 1
    ├─ Multi-Head Attention (4 heads)
    ├─ Add & Norm (residual connection)
    ├─ Feed-Forward Network (64 → 256 → 64)
    └─ Add & Norm (residual connection)
    ↓
Transformer Block 2 (same structure)
    ↓
Pooling (take first token)
    ↓
Output Projection (64 → 20 vocab)
    ↓
Softmax → Prediction
```

### Model Configuration
- **Vocabulary:** 20 tokens ([PAD], [EOS], digits 0-9, operations, syntax)
- **Embedding Dim:** 64
- **Layers:** 2 transformer blocks
- **Attention Heads:** 4 per layer
- **FFN Hidden Dim:** 256 (4× embedding dim)
- **Max Sequence Length:** 50
- **Total Parameters:** ~104,000

### Key Design Decisions

1. **Encoder-only architecture** (not decoder/autoregressive)
   - Input: full operation sequence
   - Output: single token prediction
   - Simpler than autoregressive for this task

2. **Post-Layer Normalization** (original transformer style)
   - Pattern: `x = LayerNorm(x + SubLayer(x))`
   - Works well for shallow models (2 layers)
   - Alternative Pre-LN more stable for deep models (not needed here)

3. **Sinusoidal positional encoding** (fixed, not learned)
   - Generalizes to any sequence length
   - One less thing to train
   - Standard in original transformer paper

4. **First-token pooling** for sequence representation
   - Takes `hidden[:, 0, :]` for classification
   - Alternative: mean pooling (might be better but more complex)

## Code Organization

### Module Structure
```
src/
├── layers/
│   ├── embedding.py          # Token embeddings with gradient accumulation
│   ├── linear.py             # Fully connected layers (Xavier init)
│   ├── positional_encoding.py # Sinusoidal position info (no learnable params)
│   ├── normalization.py      # Layer normalization (learnable gamma/beta)
│   ├── activations.py        # ReLU and Softmax with gradients
│   └── attention.py          # Scaled dot-product and multi-head attention
│
├── transformer.py            # FeedForward, TransformerBlock, Transformer
├── loss.py                   # CrossEntropyLoss
├── optimizer.py              # SGD and Adam (implemented from scratch)
├── train_utils.py            # train_step() and evaluate() functions
│
├── vocabluary.py             # tokenize() and detokenize()
├── data_generatpr.py         # generate_tokenized_dataset() and split_dataset()
├── data_utils.py             # create_batch() with padding
│
├── evaluation.py             # MetricsTracker, ModelCheckpoint, per-operation analysis
└── visualization.py          # Attention heatmaps and pattern detection
```

### Import Pattern
All modules use absolute imports with `sys.path.insert(0, 'src')` or `sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layers'))`.

### Critical Implementation Details

1. **Gradient Flow**
   - Every layer implements `forward()` and `backward()` manually
   - Caches intermediate values during forward for backward pass
   - Uses `get_parameters()` to return (param, grad) tuples
   - Implements `zero_grad()` to reset gradients

2. **Attention Mechanism** (src/layers/attention.py:49-165)
   - `scaled_dot_product_attention()`: Core attention operation
   - Formula: `Attention(Q,K,V) = softmax(Q@K^T / sqrt(d_k)) @ V`
   - Scaling by sqrt(d_k) prevents vanishing gradients
   - Returns both output and attention weights (for visualization)

3. **Multi-Head Attention** (src/layers/attention.py:168-599)
   - Single projection to Q/K/V, then split into heads
   - More parameter-efficient than separate projections per head
   - Reshape pattern: `(batch, seq_len, embed_dim) → (batch, num_heads, seq_len, head_dim)`
   - Backward pass is simplified (skips full softmax Jacobian for speed)

4. **Residual Connections**
   - Pattern: `x = x + SubLayer(x)` (input added to output)
   - Gradient splits into two paths: direct (through addition) and indirect (through sublayer)
   - Enables training deep networks by providing gradient highway

5. **Batching and Padding** (src/data_utils.py)
   - `create_batch()` pads sequences to max length in batch
   - Returns attention masks to ignore padding positions
   - Padding token is ID 0 ([PAD])

## Common Development Tasks

### Adding New Layer Types
1. Create class with `__init__`, `forward()`, `backward()`
2. Cache intermediate values in forward pass
3. Implement gradient computation in backward pass
4. Add `get_parameters()` and `zero_grad()` methods
5. Add comprehensive unit tests with gradient checking

### Modifying Transformer Architecture
- **Changing dimensions:** Update `embed_dim`, `num_heads`, `ffn_dim` in Transformer constructor
- **Adding layers:** Increase `num_layers` parameter (2 → 4, etc.)
- **Pre-LN vs Post-LN:** Modify TransformerBlock.forward() to swap norm/sublayer order

### Training New Models
1. Generate dataset: `generate_tokenized_dataset(num_examples, num_args=3, max_value=9)`
2. Split: `train_data, val_data, test_data = split_dataset(dataset)`
3. Create model: `Transformer(vocab_size=20, embed_dim=64, num_heads=4, num_layers=2, ffn_dim=256)`
4. Use Adam optimizer: `Adam(model.get_parameters, learning_rate=0.001)`
5. Train: `train_step(model, inputs, targets, loss_fn, optimizer)` in loop
6. Save checkpoints: `ModelCheckpoint('checkpoints/').save(model, optimizer, tracker, epoch)`

### Debugging Gradient Issues
- Tests include numerical gradient checking (finite differences vs analytical)
- Check `test_layers.py`, `test_attention.py` for examples
- Pattern: compute numerical gradient `(f(x+h) - f(x-h)) / 2h`, compare to backward()
- All gradients should match within ~1e-5 tolerance

## Important Constraints

### What This Codebase Does
- ✅ Educational transformer from scratch
- ✅ Complete forward and backward passes manually implemented
- ✅ Works with small-scale problems (toy datasets, ~100K params)
- ✅ Runs on CPU (NumPy only)
- ✅ Comprehensive documentation and tests

### What This Codebase Doesn't Do
- ❌ GPU acceleration (no CUDA/GPU)
- ❌ Modern optimizations (flash attention, gradient checkpointing, etc.)
- ❌ Large-scale training (designed for <100K examples)
- ❌ Autoregressive generation (encoder-only, not decoder)
- ❌ Dropout (mentioned in code but not implemented)
- ❌ Beam search or sampling (deterministic argmax only)

## Testing Philosophy

Each component has three types of tests:
1. **Shape tests:** Verify tensor dimensions through layers
2. **Gradient tests:** Numerical vs analytical gradient checking
3. **Functionality tests:** Verify correct computations (e.g., softmax sums to 1)

Integration test verifies model can learn (loss decreases from 2.87 → 1.17).

## Expected Results

**Medium dataset (10K examples, 20 epochs):**
- Training time: ~6 minutes
- Test accuracy: 93-99% (varies by random seed)
- All operations learnable to 97-100% with proper training

**Common pitfall:** Early implementations assumed Max/Min are "harder" operations, but actual results show all operations achieve similar accuracy (97-100%) with sufficient training. Performance differences are primarily due to random initialization.

## Vocabulary and Tokenization

**Vocabulary mapping** (src/vocabluary.py):
- Special: [PAD]=0, [EOS]=1
- Digits: '0'=2 through '9'=11
- Operations: First=12, Second=13, Last=14, Max=15, Min=16
- Syntax: '('=17, ')'=18, ','=19

**Example tokenization:**
```
"Max ( 5 3 9 )" → [15, 17, 7, 5, 11, 18]
```

## Attention Visualization

To enable real attention extraction (currently uses dummy data):
1. Modify `src/layers/attention.py` MultiHeadAttention.forward()
2. Add: `self.last_attention_weights = attention_weights` after computing attention
3. Use `extract_attention_weights()` in visualization.py

Pattern types:
- **Focused:** Low entropy, specific positions (First/Second/Last)
- **Distributed:** High entropy, all positions (Max/Min)

## File Naming Quirks

Note intentional typos in filenames (do not "fix"):
- `data_generatpr.py` (not "generator")
- `vocabluary.py` (not "vocabulary")

These were original typos kept for consistency. Code inside uses correct spelling in variable names.
