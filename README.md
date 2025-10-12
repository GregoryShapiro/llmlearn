# Transformer from Scratch

A minimal transformer implementation built from scratch using only NumPy to understand LLM architecture through a toy problem: learning simple operations on digits.

## Project Goal

Build a transformer that learns to execute simple operations:
- **First(a, b, c)** → returns a
- **Second(a, b, c)** → returns b
- **Last(a, b, c)** → returns c
- **Max(a, b, c)** → returns maximum
- **Min(a, b, c)** → returns minimum

## Current Status

### ✅ Completed (Phase 1 - Data Pipeline)

1. **Vocabulary & Tokenization** (`src/vocabluary.py`)
   - 20-token vocabulary with special tokens, digits, operations, and syntax
   - Tokenization and detokenization functions
   - Multi-digit number handling

2. **Data Generation** (`src/data_generatpr.py`)
   - Random example generation with balanced operations
   - Dataset creation and train/val/test splitting
   - Operation distribution analysis

3. **Data Batching** (`src/data_utils.py`)
   - Sequence padding and batching
   - Attention mask creation
   - DataLoader with shuffling support

4. **Core Layers** (`src/layers.py`) - PARTIAL
   - Embedding layer (complete with forward/backward)
   - Linear layer (complete with forward/backward)
   - Numerical gradient checking

### 🚧 In Progress (Phase 2)

- Positional Encoding
- Layer Normalization
- Activation Functions (ReLU, Softmax)

### 📋 Next Steps

- Multi-head attention mechanism
- Transformer blocks
- Training loop and optimization
- Evaluation and visualization

## Architecture

```
Vocabulary Size:    20 tokens
Embedding Dim:      64
Layers:             2 transformer blocks
Attention Heads:    4 per layer
FFN Hidden Dim:     256
Max Sequence Len:   50
Total Parameters:   ~104,000
```

## Usage

Run individual module tests:
```bash
python src/vocabluary.py
python src/data_generatpr.py
python src/data_utils.py
python src/layers.py
```

## Dependencies

- Python 3.8+
- NumPy

Optional (for visualization):
- Matplotlib
- Seaborn

## Documentation

- `design,md` - Complete design document
- `tasks.md` - Detailed task breakdown and progress tracking
