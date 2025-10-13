# Manual Model Testing Guide

This guide shows you how to test your trained model manually with your own examples.

## What's Been Saved

After training, you have two trained models:

1. **`checkpoints/best_model_run1.pkl`** - First training run (seed=42)
   - Test Accuracy: 93.5%
   - Best operation: First (97.4%)
   - Hardest operation: Second (89.9%)

2. **`checkpoints/best_model_run2.pkl`** - Second training run (seed=123)
   - Test Accuracy: 99.2% ⭐
   - Best operation: Last & Min (100.0%)
   - Hardest operation: First (97.5%)

## Quick Start - Test The Model

### Option 1: Interactive Mode (Best for Manual Testing)

```bash
python3 test_model_manually.py
```

Then select option `1` for interactive mode.

**Example session:**
```
Operation (First/Second/Last/Max/Min): Max
Number 1 (0-9): 7
Number 2 (0-9): 3
Number 3 (0-9): 9

Testing: Max(7, 3, 9)
======================================================================
Predicted: 9
Correct:   9
Result:    ✓ CORRECT
Confidence: 99.42%

Top 5 Predictions:
  1. 9         99.42%  ←
  2. 7          0.35%
  3. 3          0.14%
  4. 8          0.04%
  5. 1          0.02%
```

### Option 2: Batch Test Mode (Test Multiple Examples)

```bash
python3 test_model_manually.py
```

Then select option `2` for batch test mode. This will test 14 predefined examples including edge cases.

### Option 3: Quick Single Test

```bash
python3 test_model_manually.py
```

Select option `3` for a quick test of `Max(7, 3, 9)`.

## Advanced Usage - Python API

You can also use the model programmatically in your own scripts:

```python
import sys
sys.path.insert(0, 'src')

import numpy as np
import pickle
from transformer import Transformer
from vocabluary import tokenize_with_numbers, detokenize

# Load model
with open('checkpoints/best_model_run2.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

model = Transformer(
    vocab_size=20,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    ffn_dim=256,
    max_seq_len=50
)

# Load parameters
model_params = checkpoint['model_params']
saved_params = model.get_parameters()
for (param, grad), saved_param in zip(saved_params, model_params):
    param[:] = saved_param

# Test an example: Max(5, 8, 3)
input_sequence = ['Max', '(', 5, ',', 8, ',', 3, ')']
input_indices = tokenize_with_numbers(input_sequence)
input_padded = input_indices + [0] * (20 - len(input_indices))
input_batch = np.array(input_padded).reshape(1, -1)

# Get prediction
predictions = model.forward(input_batch)
predicted_idx = np.argmax(predictions[0])
predicted_token = detokenize([predicted_idx])[0]

print(f"Max(5, 8, 3) = {predicted_token}")
print(f"Confidence: {predictions[0][predicted_idx]:.2%}")
```

## What's in the Checkpoint File

The `.pkl` files contain:

```python
checkpoint = {
    'model_params': [...],      # All trained weights (102,036 parameters)
    'optimizer_state': {...},   # Adam optimizer state (momentum, velocity)
    'history': {                # Training history
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...]
    },
    'epoch': 17,               # Epoch number when saved
    'best_val_acc': 0.994      # Best validation accuracy achieved
}
```

## Testing Different Models

To test the first training run instead of the second:

```python
# In test_model_manually.py, change line 15:
model = load_trained_model('checkpoints/best_model_run1.pkl')  # First run
```

Or:

```python
model = load_trained_model('checkpoints/best_model_run2.pkl')  # Second run (default)
```

## Test Cases to Try

### Easy Cases
- `First(5, 3, 8)` → 5
- `Second(2, 7, 1)` → 7
- `Last(4, 6, 9)` → 9
- `Max(3, 8, 2)` → 8
- `Min(7, 1, 5)` → 1

### Edge Cases
- **All same numbers:** `Max(5, 5, 5)` → 5
- **Duplicates:** `Max(9, 2, 9)` → 9
- **Zeros:** `Min(0, 5, 0)` → 0
- **Consecutive:** `Max(7, 8, 9)` → 9

### Potential Failure Cases
- `Second(4, 4, 7)` - Two numbers are the same
- `Min(9, 0, 9)` - Minimum with duplicates
- `Last(1, 4, 0)` - Zero at the end

## Comparing Two Training Runs

You can compare both models on the same examples:

```python
# Load both models
model1 = load_trained_model('checkpoints/best_model_run1.pkl')
model2 = load_trained_model('checkpoints/best_model_run2.pkl')

# Test same example on both
test_cases = [('Max', 7, 3, 9), ('Second', 5, 5, 8)]

for op, n1, n2, n3 in test_cases:
    pred1, conf1, _ = predict(model1, op, n1, n2, n3)
    pred2, conf2, _ = predict(model2, op, n1, n2, n3)

    print(f"{op}({n1}, {n2}, {n3}):")
    print(f"  Model 1: {pred1} ({conf1:.2%})")
    print(f"  Model 2: {pred2} ({conf2:.2%})")
```

## Understanding the Output

When you test an example, you'll see:

- **Predicted:** What the model thinks the answer is
- **Correct:** The actual correct answer
- **Result:** ✓ CORRECT or ✗ WRONG
- **Confidence:** How sure the model is (0-100%)
- **Top 5 Predictions:** The model's top 5 guesses with probabilities

**High confidence (>95%)** = Model is very sure
**Low confidence (<80%)** = Model is uncertain, might be wrong

## Troubleshooting

### Error: "File not found"
Make sure you're in the `/home/grisha/dev/llmlearn` directory:
```bash
cd /home/grisha/dev/llmlearn
python3 test_model_manually.py
```

### Error: "Module not found"
The script adds `src/` to the path automatically. If it still fails:
```bash
export PYTHONPATH="${PYTHONPATH}:src"
python3 test_model_manually.py
```

### Want to test the large dataset model?
After training the 100K dataset, the checkpoint will be saved as `checkpoints/best_model.pkl` and you can test it the same way.

## Next Steps

1. ✅ Test both trained models (run1 and run2)
2. ✅ Compare their predictions on tricky examples
3. ✅ Find edge cases where they disagree
4. 🔄 Train on large dataset (100K examples) for even better performance
5. 🔄 Test the large model to see if it achieves 95%+ on all operations

---

**Have fun testing your transformer!** 🚀
