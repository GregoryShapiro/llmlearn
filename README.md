# Transformer from Scratch

A complete transformer implementation built from scratch using only NumPy to understand LLM architecture through a toy problem: learning simple operations on digits.

## Project Goal

Build a transformer that learns to execute simple operations:
- **First(a, b, c)** → returns a
- **Second(a, b, c)** → returns b
- **Last(a, b, c)** → returns c
- **Max(a, b, c)** → returns maximum
- **Min(a, b, c)** → returns minimum

**Example:**
```
Input:  "Max ( 5 3 9 )"
Output: "9"

Input:  "First ( 2 8 4 )"
Output: "2"
```

## Current Status

### ✅ **All Core Phases Complete!**

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Data Pipeline (vocab, generation, batching) |
| **Phase 2** | ✅ Complete | Core Components (embedding, linear, normalization) |
| **Phase 3** | ✅ Complete | Attention Mechanism (multi-head attention) |
| **Phase 4** | ✅ Complete | Transformer Architecture (blocks, full model) |
| **Phase 5** | ✅ Complete | Training Infrastructure (loss, optimizer, training loop) |
| **Phase 6** | ✅ Complete | Evaluation & Visualization (metrics, attention viz) |

**Integration Tests:** ✅ All passing - model learns (loss decreases from 2.87 → 1.17)

### 🎯 **Trained Models Available!**

Multiple pre-trained models are ready to test:

**Single-Digit Models (0-9):**
- **Run 1**: 93.5% test accuracy (seed=42)
- **Run 2**: 99.2% test accuracy (seed=123) ⭐ Best for single digits

**Double-Digit Model (0-99):**
- **best_model_double_digits.pkl**: 83.7% test accuracy - handles numbers 0-99!

See [QUICK_START_TESTING.md](QUICK_START_TESTING.md) for how to use them.
See [DOUBLE_DIGIT_TRAINING.md](DOUBLE_DIGIT_TRAINING.md) for double-digit training details.

## Architecture

```
Input Tokens → Embedding → Positional Encoding
                               ↓
                    Transformer Block 1
                    ├─ Multi-Head Attention (4 heads)
                    ├─ Add & Norm
                    ├─ Feed-Forward Network
                    └─ Add & Norm
                               ↓
                    Transformer Block 2
                    ├─ Multi-Head Attention (4 heads)
                    ├─ Add & Norm
                    ├─ Feed-Forward Network
                    └─ Add & Norm
                               ↓
                    Output Projection → Softmax
                               ↓
                         Predicted Token
```

**Model Configuration:**
```
Vocabulary Size:    20 tokens
Embedding Dim:      64
Layers:             2 transformer blocks
Attention Heads:    4 per layer
FFN Hidden Dim:     256
Max Sequence Len:   50
Total Parameters:   ~104,000
```

## Quick Start

### 1. Installation

```bash
# Required
pip install numpy

# Optional (for visualization)
pip install matplotlib
```

### 2. Run Tests

```bash
# Test individual components
python3 tests/test_layers.py
python3 tests/test_attention.py
python3 tests/test_transformer.py
python3 tests/test_training.py

# Run full integration test
python3 tests/test_integration.py
```

### 3. Train the Model

**Quick Option:** Use the interactive training script:
```bash
# Train with step-by-step visualization (small dataset ~1K examples)
python3 train_step_by_step.py --size small --epochs 10

# Train with medium dataset (~10K examples)
python3 train_step_by_step.py --size medium --epochs 20

# Train with large dataset (~100K examples) - takes 2-3 hours
python3 train_step_by_step.py --size large --epochs 50 --no-interactive

# Train on double-digit numbers (0-99 range)
python3 train_double_digits.py
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.
See [DOUBLE_DIGIT_TRAINING.md](DOUBLE_DIGIT_TRAINING.md) for double-digit training details.

**Manual Option:** Create a custom training script (e.g., `train.py`):

```python
import sys
sys.path.insert(0, 'src')

import numpy as np
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

from vocabluary import VOCAB_SIZE
from data_generatpr import generate_tokenized_dataset, split_dataset
from data_utils import create_batch
from transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from train_utils import train_step, evaluate
from evaluation import MetricsTracker, ModelCheckpoint

# Generate dataset
print("Generating dataset...")
dataset = generate_tokenized_dataset(
    num_examples=10000,
    num_args=3,
    max_value=9,
    balance_operations=True
)

# Split into train/val/test
train_data, val_data, test_data = split_dataset(dataset)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Create model
print("\nCreating model...")
model = Transformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    ffn_dim=256,
    max_seq_len=50
)

# Training setup
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.get_parameters, learning_rate=0.001)
tracker = MetricsTracker()
checkpoint = ModelCheckpoint('checkpoints/')

# Prepare data batches
print("\nPreparing data...")
train_inputs, train_targets, _ = create_batch(train_data, max_length=20)
train_targets = train_targets[:, 0]  # Take first digit only

val_inputs, val_targets, _ = create_batch(val_data, max_length=20)
val_targets = val_targets[:, 0]

# Training loop
print("\nStarting training...")
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 60)

    # Shuffle training data
    shuffle_idx = np.random.permutation(len(train_inputs))
    train_inputs_shuffled = train_inputs[shuffle_idx]
    train_targets_shuffled = train_targets[shuffle_idx]

    # Train for one epoch
    num_batches = (len(train_inputs) + batch_size - 1) // batch_size
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(train_inputs))

        batch_inputs = train_inputs_shuffled[start:end]
        batch_targets = train_targets_shuffled[start:end]

        loss, acc = train_step(model, batch_inputs, batch_targets, loss_fn, optimizer)
        epoch_loss += loss * len(batch_inputs)
        epoch_acc += acc * len(batch_inputs)

    # Compute epoch averages
    train_loss = epoch_loss / len(train_inputs)
    train_acc = epoch_acc / len(train_inputs)

    # Evaluate on validation set
    val_loss, val_acc = evaluate(model, val_inputs, val_targets, loss_fn, batch_size=batch_size)

    # Update metrics
    tracker.update(train_loss, train_acc, val_loss, val_acc)
    tracker.print_summary()

    # Save best model
    if tracker.is_best_epoch():
        checkpoint.save(model, optimizer, tracker, epoch)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
tracker.print_summary()

# Test on test set
print("\nEvaluating on test set...")
test_inputs, test_targets, _ = create_batch(test_data, max_length=20)
test_targets = test_targets[:, 0]
test_loss, test_acc = evaluate(model, test_inputs, test_targets, loss_fn, batch_size=batch_size)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```

Run the training:
```bash
python3 train.py
```

### 4. Test Trained Models

Test the pre-trained models interactively:
```bash
python3 test_model_manually.py
```

Then select:
- **Option 1**: Interactive mode - test your own examples
- **Option 2**: Batch test mode - test predefined examples
- **Option 3**: Quick single test

See [MANUAL_TESTING.md](MANUAL_TESTING.md) for detailed testing instructions.

### 5. Visualize Results

#### Plot Training Curves

```python
from evaluation import MetricsTracker, plot_training_curves

# Load from checkpoint or use existing tracker
plot_training_curves(tracker, save_path='training_curves.png')
```

#### Analyze Per-Operation Performance

```python
from evaluation import compute_per_operation_accuracy, print_operation_analysis

operation_names = ['First', 'Second', 'Last', 'Max', 'Min']
print_operation_analysis(model, test_data, operation_names)
```

Expected output:
```
Per-Operation Analysis:
============================================================
  Max       : ███████████████████████████████░░░░░░░░░ 78.5%
  Min       : ████████████████████████████████░░░░░░░░ 82.3%
  Second    : ████████████████████████████████████░░░░ 92.1%
  Last      : █████████████████████████████████████░░░ 94.7%
  First     : ██████████████████████████████████████░░ 95.2%

  Average Accuracy: 88.6%
  Hardest: Max (78.5%)
  Easiest: First (95.2%)
```

#### Visualize Attention Patterns

```python
from visualization import extract_attention_weights, plot_attention_heatmap, visualize_attention_pattern
from vocabluary import detokenize

# Pick an example
example = test_data[0]
input_indices, answer_indices = example

# Prepare input
input_padded = np.array(input_indices + [0] * (20 - len(input_indices)))
input_batch = input_padded.reshape(1, -1)

# Extract attention (currently uses dummy data - see note below)
attention_weights = extract_attention_weights(model, input_batch)

# Get token names for display
token_names = [detokenize([idx]) for idx in input_indices]

# Text-based visualization (no dependencies needed)
visualize_attention_pattern(
    attention_weights,
    input_batch,
    token_names,
    head_idx=0,
    layer_idx=0
)

# Or use matplotlib for prettier plots
plot_attention_heatmap(
    attention_weights,
    input_batch,
    token_names,
    layer_idx=0,
    head_idx=0,
    save_path='attention_head_0.png'
)
```

**Note:** To enable real attention extraction, modify `src/layers/attention.py`:
```python
# In MultiHeadAttention.forward(), add:
self.last_attention_weights = attention_weights  # Save for visualization
```

## Project Structure

```
llmlearn/
├── lessons/                 # 📚 9-lesson educational series (START HERE!)
│   ├── README.md            # Lesson overview and learning path
│   ├── lesson_01_*.md       # Embeddings & positional encoding
│   ├── lesson_02_*.md       # Attention mechanism
│   ├── lesson_03_*.md       # Residual connections, LayerNorm, FFN
│   ├── lesson_04_*.md       # Transformer block stacking
│   ├── lesson_05_*.md       # Output projection
│   ├── lesson_06_*.md       # Softmax & loss calculation
│   ├── lesson_07_*.md       # Backpropagation
│   ├── lesson_08_*.md       # Training loop & optimizers
│   └── lesson_09_*.md       # Training dynamics over time
├── src/
│   ├── layers/              # Neural network components
│   │   ├── embedding.py     # Token embeddings
│   │   ├── linear.py        # Fully connected layers
│   │   ├── positional_encoding.py  # Position information
│   │   ├── normalization.py # Layer normalization
│   │   ├── activations.py   # ReLU and Softmax
│   │   └── attention.py     # Multi-head attention
│   ├── transformer.py       # Complete transformer model
│   ├── transformer_decoder.py # GPT-style decoder variant
│   ├── loss.py              # Cross-entropy loss
│   ├── optimizer.py         # SGD and Adam optimizers
│   ├── train_utils.py       # Training loop utilities
│   ├── vocabluary.py        # Vocabulary and tokenization
│   ├── data_generatpr.py    # Training data generation
│   ├── data_utils.py        # Batching and padding
│   ├── evaluation.py        # Metrics tracking and checkpointing
│   └── visualization.py     # Attention visualization
├── tests/
│   ├── test_layers.py       # Core layer tests
│   ├── test_attention.py    # Attention mechanism tests
│   ├── test_transformer.py  # Full model tests
│   ├── test_training.py     # Training infrastructure tests
│   └── test_integration.py  # End-to-end integration test
├── train_step_by_step.py    # Interactive training script
├── train_double_digits.py   # Double-digit training (0-99)
├── test_model_manually.py   # Manual model testing tool
├── design.md                # Architecture and design decisions
├── tasks.md                 # Task breakdown and progress
├── TRAINING_GUIDE.md        # Detailed training instructions
├── DOUBLE_DIGIT_TRAINING.md # Double-digit training guide
├── MANUAL_TESTING.md        # Testing guide
├── QUICK_START_TESTING.md   # Quick testing reference
└── README.md                # This file
```

## Key Features

### Built from Scratch
- **No PyTorch/TensorFlow** - only NumPy
- Every component implemented manually
- Complete backpropagation implementation
- Educational focus with extensive documentation

### Comprehensive Testing
- Unit tests for all components
- Gradient checking (numerical vs analytical)
- Integration tests verify end-to-end functionality
- Tests confirm model learns (loss decreases)

### Evaluation Tools
- **MetricsTracker**: Track training progress, detect overfitting
- **ModelCheckpoint**: Save/load best models
- **Per-Operation Analysis**: Identify which operations are hardest
- **Training Curves**: Visualize loss and accuracy over time

### Attention Visualization
- Text-based heatmaps (no dependencies)
- Matplotlib visualizations (optional)
- Automatic pattern detection (focused vs distributed)
- Multi-head comparison views

## Learning Resources

### 📚 Complete Lesson Series

**New!** This project includes a **comprehensive 9-lesson series** that takes you from basic concepts to complete understanding of transformer architecture.

👉 **Start here:** [lessons/README.md](lessons/README.md)

**Quick overview of lessons:**

| # | Lesson | Topics | Time |
|---|--------|--------|------|
| 01 | [Embeddings & Positional Encoding](lessons/lesson_01_embeddings_and_positional_encoding.md) | Token embeddings, sinusoidal encoding | 2h |
| 02 | [Attention Mechanism](lessons/lesson_02_attention_mechanism.md) | Scaled dot-product, multi-head attention | 2.5h |
| 03 | [Residual, LayerNorm & FFN](lessons/lesson_03_residual_layernorm_ffn.md) | Add & Norm pattern, feed-forward networks | 2h |
| 04 | [Transformer Block 2](lessons/lesson_04_transformer_block_2.md) | Stacking blocks, hierarchical learning | 1h |
| 05 | [Output Projection](lessons/lesson_05_output_projection.md) | From representations to predictions | 1h |
| 06 | [Softmax & Loss](lessons/lesson_06_softmax_and_loss.md) | Probability distributions, cross-entropy | 1h |
| 07 | [Backpropagation](lessons/lesson_07_backpropagation.md) | Gradients through all layers | 2h |
| 08 | [Training Loop](lessons/lesson_08_training_loop.md) | SGD, Adam optimizer, mini-batches | 1.5h |
| 09 | [Training Dynamics](lessons/lesson_09_training_dynamics.md) | How the model learns over time | 2h |

**Total:** ~15 hours of in-depth material covering 12 stages of the transformer pipeline.

### Key Concepts Explained

1. **Embeddings** - How tokens become vectors
2. **Positional Encoding** - Why position matters
3. **Attention** - How transformers "look at" different parts of input
4. **Multi-Head Attention** - Learning multiple relationship types
5. **Layer Normalization** - Stabilizing deep network training
6. **Residual Connections** - Enabling gradient flow
7. **Adam Optimizer** - Adaptive learning rates per parameter

### Additional Documentation

- **[lessons/](lessons/)** - Complete 9-lesson educational series (recommended starting point)
- **[design.md](design.md)** - Architecture decisions and design rationale
- **[CLAUDE.md](CLAUDE.md)** - Guide for AI assistants working with this codebase
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training instructions
- **[MANUAL_TESTING.md](MANUAL_TESTING.md)** - Interactive model testing guide
- **[HOW_REAL_LLMS_WORK.md](HOW_REAL_LLMS_WORK.md)** - Comparison to production LLMs (GPT, BERT)

## Expected Results

### Actual Results from Training Runs

**Medium Dataset (10,000 examples, 20 epochs):**

**Run 1 (seed=42):**
- **Test Accuracy**: 93.5%
- **Training Time**: ~6.2 minutes
- Best Operation: First (97.4%)
- Hardest Operation: Second (89.9%)

**Run 2 (seed=123) ⭐:**
- **Test Accuracy**: 99.2%
- **Training Time**: ~6.2 minutes
- **Validation Accuracy**: 99.4%
- Best Operations: Last & Min (100.0%)
- Hardest Operation: First (97.5%)

**Key Insight:** Accuracy differences between runs are due to random initialization, not inherent operation difficulty. All operations can achieve 97-100% with proper training.

**Expected for Large Dataset (100,000 examples):**
- **Test Accuracy**: 95%+ on all operations
- **Training Time**: 2-3 hours
- **Confidence**: 99%+ on most predictions

## Troubleshooting

### Model Not Learning
- Check learning rate (try 0.001 to 0.01)
- Verify gradients are flowing (check test_integration.py)
- Ensure data is balanced across operations
- Try smaller model first (1 layer, 2 heads)

### Overfitting
- Reduce model size
- Use more training data
- Check train vs validation accuracy gap
- Stop training when validation accuracy peaks

### Attention Visualization Not Working
- Modify `MultiHeadAttention.forward()` to save attention weights
- Ensure matplotlib is installed for plots
- Use text-based visualization as fallback

## Dependencies

**Required:**
- Python 3.8+
- NumPy

**Optional:**
- Matplotlib (for plotting)

## Development Timeline

This project was built in phases:
- **Phase 1**: Data Pipeline (~3 hours)
- **Phase 2**: Core Components (~6 hours)
- **Phase 3**: Attention Mechanism (~6 hours)
- **Phase 4**: Transformer Architecture (~3 hours)
- **Phase 5**: Training Infrastructure (~6 hours)
- **Phase 6**: Evaluation & Visualization (~4 hours)

**Total**: ~28 hours from scratch to fully functional transformer

## Future Extensions

Possible enhancements (not implemented):
1. **Nested Operations**: `Max(First(2,8), Second(5,1))`
2. **Arithmetic**: `Add(Max(3,7), Min(2,9))`
3. **Longer Sequences**: 5-10 arguments instead of 3
4. **Autoregressive Mode**: Generate multi-digit answers
5. **PyTorch Comparison**: Verify implementation correctness

## License

Educational project - free to use and modify.

## Acknowledgments

Built to understand transformer architecture through implementation. Inspired by "Attention is All You Need" (Vaswani et al., 2017).

---

**Ready to train your transformer!** 🚀

Questions or issues? Check:
- `design.md` - Architecture details
- `tasks.md` - Implementation progress
- `tests/` - Working examples
- `TRAINING_GUIDE.md` - Training instructions
- `MANUAL_TESTING.md` - Testing guide
