# How to Run Manual Tests - Updated Guide

## Quick Start

Run the testing script:
```bash
python3 test_model_manually.py
```

Then select **Option 1** for Interactive Mode.

---

## New Interactive Mode Features ✨

### Single-Line Input Format

You can now enter complete expressions in one line:

```
> Max(23, 212, 11)
> First(5, 8, 3)
> Min(0, 9, 4)
```

### Supported Input Formats

All of these work:
```
Max(5,8,3)          # No spaces
Max(5, 8, 3)        # With spaces
Max (5, 8, 3)       # Space before parenthesis
first(1,2,3)        # Lowercase (auto-capitalized)
MIN(9,4,7)          # Uppercase (auto-capitalized)
```

### Examples

```bash
$ python3 test_model_manually.py
======================================================================
TRAINED MODEL MANUAL TESTING
======================================================================
Loading model from checkpoints/best_model.pkl...
✓ Model loaded successfully!
  Best validation accuracy: 97.30%
  Training epoch: 20

======================================================================
TESTING MODES
======================================================================
1. Interactive Mode - Test your own examples
2. Batch Test Mode - Test predefined examples
3. Quick Test - Single example
4. Exit

Select mode (1-4): 1

======================================================================
INTERACTIVE MODE
======================================================================

Test your model interactively!

Usage:
  Enter expressions in the format: Operation(num1, num2, num3)

Examples:
  Max(5, 8, 3)
  First(23, 212, 11)
  Min(0, 9, 4)

Supported operations: First, Second, Last, Max, Min
Type 'quit' or 'exit' to exit

----------------------------------------------------------------------

> Max(23, 212, 11)

======================================================================
Testing: Max(23, 212, 11)
======================================================================

Predicted: 2
Correct:   212
Result:    ✗ WRONG
Confidence: 45.23%

Top 5 Predictions:
  1. 2         45.23%
  2. 1         23.45%
  3. 3         12.34%
  4. 5          8.90%
  5. 8          4.56%

> Max(5, 8, 3)

======================================================================
Testing: Max(5, 8, 3)
======================================================================

Predicted: 8
Correct:   8
Result:    ✓ CORRECT
Confidence: 99.42%

Top 5 Predictions:
  1. 8         99.42%  ←
  2. 5          0.35%
  3. 3          0.14%
  4. 7          0.04%
  5. 9          0.02%

> quit
Exiting...
```

---

## Why Multi-Digit Numbers May Fail

**Important Note:** The model was trained on **single-digit numbers only (0-9)**.

When you input multi-digit numbers like `Max(23, 212, 11)`:
- The tokenizer splits them into individual digits: `2`, `3`, `2`, `1`, `2`, `1`, `1`
- The model sees: `Max(2, 3, 2, 1, 2, 1, 1)` - 7 arguments instead of 3!
- This is **out of distribution** - the model never saw this pattern during training
- Results will be unpredictable

**Expected behavior:**
- ✅ **Single digits (0-9)**: Model performs well (93-99% accuracy)
- ❌ **Multi-digit numbers**: Model will likely fail or give random predictions

**Example:**
```
Max(5, 8, 3)      → Works great! ✓
Max(23, 212, 11)  → Won't work correctly ✗
```

---

## Testing Within Training Distribution

To test the model properly, use single-digit numbers:

```
> Max(5, 8, 3)
> First(0, 9, 4)
> Min(7, 1, 5)
> Second(2, 8, 6)
> Last(3, 5, 9)
```

---

## Error Messages

### Invalid Format
```
> Max(5, 8)
❌ Invalid format!
   Expected: Operation(num1, num2, num3)
   Example: Max(5, 8, 3)
```

### Invalid Operation
```
> Average(5, 8, 3)
❌ Invalid operation: 'Average'
   Supported: First, Second, Last, Max, Min
```

---

## Alternative Models

To test a different checkpoint, edit line 17 in `test_model_manually.py`:

```python
# Current (most recent training)
def load_trained_model(checkpoint_path='checkpoints/best_model.pkl'):

# Change to test Run 2 (99.2% accuracy)
def load_trained_model(checkpoint_path='checkpoints/best_model_run2.pkl'):

# Or test Run 1 (93.5% accuracy)
def load_trained_model(checkpoint_path='checkpoints/best_model_run1.pkl'):
```

---

## Quick Command Reference

| Action | Input |
|--------|-------|
| Test example | `Max(5, 8, 3)` |
| Exit | `quit`, `exit`, or `q` |
| Skip input | Just press Enter |

---

## Other Testing Modes

### Batch Test Mode (Option 2)
Tests 14 predefined examples including edge cases:
- All same numbers: `Max(5, 5, 5)`
- Duplicates: `Max(9, 2, 9)`
- Zeros: `Min(0, 5, 0)`

### Quick Test (Option 3)
Runs a single test: `Max(7, 3, 9)`

---

**Updated:** 2025-10-20
**Script:** `test_model_manually.py`
**Current Model:** `checkpoints/best_model.pkl` (93.5% test accuracy)
