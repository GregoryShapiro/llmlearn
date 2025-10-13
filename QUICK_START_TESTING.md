# Quick Start - Test Your Trained Model

## Your Trained Models

You have **2 trained models** ready to test:

### Model 1 (Run 1 - Seed 42)
- **File:** `checkpoints/best_model_run1.pkl`
- **Test Accuracy:** 93.5%
- **Training Time:** ~6.2 minutes

### Model 2 (Run 2 - Seed 123) ⭐ BEST
- **File:** `checkpoints/best_model_run2.pkl`
- **Test Accuracy:** 99.2%
- **Training Time:** ~6.2 minutes
- **Validation Accuracy:** 99.4%

---

## How to Test Manually

### 1. Interactive Mode (Recommended)

```bash
python3 test_model_manually.py
```

Select option `1`, then test your own examples:

```
Operation (First/Second/Last/Max/Min): Max
Number 1 (0-9): 7
Number 2 (0-9): 3
Number 3 (0-9): 9
```

Output:
```
Testing: Max(7, 3, 9)
======================================================================
Predicted: 9
Correct:   9
Result:    ✓ CORRECT
Confidence: 99.98%
```

---

### 2. Quick Python Test

Create a file `my_test.py`:

```python
import sys
sys.path.insert(0, 'src')

import numpy as np
import pickle
from transformer import Transformer
from vocabluary import tokenize_with_numbers, detokenize

# Load the best model
with open('checkpoints/best_model_run2.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

model = Transformer(
    vocab_size=20, embed_dim=64, num_heads=4,
    num_layers=2, ffn_dim=256, max_seq_len=50
)

# Load trained weights
for i, (param, grad) in enumerate(model.get_parameters()):
    param[:] = checkpoint['model_params'][f'param_{i}']

print(f"Model loaded! Val accuracy: {checkpoint['best_val_acc']:.2%}")

# Test: Max(5, 8, 3)
input_seq = ['Max', '(', 5, ',', 8, ',', 3, ')']
input_indices = tokenize_with_numbers(input_seq)
input_padded = input_indices + [0] * (20 - len(input_indices))

predictions = model.forward(np.array(input_padded).reshape(1, -1))
predicted = detokenize([np.argmax(predictions[0])])[0]

print(f"\nMax(5, 8, 3) = {predicted}")
print(f"Confidence: {predictions[0][np.argmax(predictions[0])]:.2%}")
```

Run it:
```bash
python3 my_test.py
```

---

## Test Results from Run 2

The model achieves near-perfect accuracy:

| Operation | Accuracy | Examples |
|-----------|----------|----------|
| **First** | 97.5% | First(5, 2, 8) = 5 |
| **Second** | 99.0% | Second(1, 7, 9) = 7 |
| **Last** | 100.0% | Last(4, 6, 9) = 9 |
| **Max** | 99.4% | Max(3, 8, 2) = 8 |
| **Min** | 100.0% | Min(7, 1, 5) = 1 |

**Overall: 99.2% test accuracy**

---

## What You Can Do

✅ Test your own examples interactively
✅ Compare both trained models (run1 vs run2)
✅ Find edge cases where the model might fail
✅ Examine confidence scores for predictions
✅ See top-5 predictions for each example

---

## Files Saved

```
checkpoints/
├── best_model_run1.pkl    # First training (93.5% accuracy)
└── best_model_run2.pkl    # Second training (99.2% accuracy) ⭐

training_medium_run1.log   # Full training log (run 1)
training_medium_run2.log   # Full training log (run 2)

test_model_manually.py     # Interactive testing script
```

---

## Next Step: Train on 100K Dataset

For even better performance, train on the large dataset:

```bash
python3 train_step_by_step.py --size large --epochs 50 --no-interactive
```

Expected results:
- **Time:** 2-3 hours
- **Test Accuracy:** 95%+ on all operations
- **Confidence:** 99%+ on most predictions

---

**Your transformer is trained and ready to use!** 🎉

For detailed documentation, see `MANUAL_TESTING.md`
