# Step-by-Step Training Guide

## Quick Start

### Option 1: Small Dataset (1,000 examples) - RECOMMENDED FOR FIRST RUN
```bash
python3 train_step_by_step.py --size small --epochs 10
```
**Time:** ~2-3 minutes
**Purpose:** Quick test to see the model learn

### Option 2: Medium Dataset (10,000 examples)
```bash
python3 train_step_by_step.py --size medium --epochs 20
```
**Time:** ~15-20 minutes
**Purpose:** Realistic training, good accuracy

### Option 3: Large Dataset (100,000 examples)
```bash
python3 train_step_by_step.py --size large --epochs 50 --no-interactive
```
**Time:** ~2-3 hours
**Purpose:** Best possible accuracy
**Note:** Use `--no-interactive` flag to skip pauses (useful for long training runs)

## What You'll See

The script will guide you through 10 steps with interactive prompts:

### Step 1: Dataset Generation
- Shows how many examples are being created
- Operation distribution (balanced across First, Second, Last, Max, Min)
- Sequence length statistics
- Sample examples with token visualization

### Step 2: Train/Val/Test Split
- 80% training, 10% validation, 10% test
- Shows exact counts for each split

### Step 3: Data Batching
- Creates batches with padding
- Shows batch shapes and example batches
- Attention masks for padded positions

### Step 4: Model Architecture
- Complete model structure
- Parameter count breakdown
- Layer-by-layer details

### Step 5: Forward Pass (Before Training)
- Run untrained model on example
- See random predictions (model hasn't learned yet!)
- Top 5 prediction probabilities

### Step 6: Training Setup
- Optimizer configuration (Adam)
- Learning rate, batch size, epochs
- Total number of parameter updates

### Step 7: Training Loop
- Progress bar for each epoch
- Batch-level updates
- Train and validation metrics
- Example predictions every 2 epochs
- Automatic checkpoint saving when validation improves

### Step 8: Test Set Evaluation
- Final accuracy on held-out test set
- Per-operation accuracy breakdown
- Visual progress bars showing which operations are easy/hard

### Step 9: Sample Predictions
- Detailed predictions for each operation type
- Top 5 predictions with probabilities
- See if model got it right or wrong

### Step 10: Attention Visualization
- Attention pattern heatmap (text-based)
- Pattern analysis (focused vs distributed)
- Entropy calculation

## Expected Results

### Small Dataset (1,000 examples, 10 epochs)
```
Train Accuracy: 60-70%
Val Accuracy:   55-65%
Test Accuracy:  55-65%
Time:           2-3 minutes

Per-Operation:
  First:  70-80%
  Second: 65-75%
  Last:   70-80%
  Max:    40-50% (hardest)
  Min:    45-55%
```

### Medium Dataset (10,000 examples, 20 epochs)
```
Train Accuracy: 93-99%
Val Accuracy:   93-99%
Test Accuracy:  93-99%
Time:           ~6 minutes

Actual Results (from training runs):
  Run 1 (seed=42):  93.5% test accuracy
  Run 2 (seed=123): 99.2% test accuracy ⭐

Per-Operation (Run 2):
  First:  97.5%
  Second: 99.0%
  Last:   100.0%
  Max:    99.4%
  Min:    100.0%
```

**Note:** Performance varies with random seed. Both runs show the model can learn all operations very well (93%+ overall).

### Large Dataset (100,000 examples, 50 epochs)
```
Train Accuracy: 95%+
Val Accuracy:   90-95%
Test Accuracy:  90-95%
Time:           2-3 hours

Per-Operation:
  First:  98%+
  Second: 95%+
  Last:   98%+
  Max:    85-90%
  Min:    85-90%
```

## Advanced Options

### Custom Configuration
```bash
# Custom epochs and batch size
python3 train_step_by_step.py --size medium --epochs 30 --batch-size 64

# Custom learning rate
python3 train_step_by_step.py --size small --lr 0.01

# Faster training (larger batches, higher LR)
python3 train_step_by_step.py --size medium --epochs 15 --batch-size 128 --lr 0.005
```

### All Available Options
```
--size           Dataset size: small (1K), medium (10K), large (100K)
--epochs         Number of training epochs (default: 10)
--batch-size     Batch size (default: 32)
--lr             Learning rate (default: 0.001)
--no-interactive Skip interactive pauses (useful for long runs)
```

## Interactive Features

The script pauses at key points so you can:
- ✅ Read the output carefully
- ✅ Understand what's happening
- ✅ Take screenshots or notes
- ✅ Press Enter when ready to continue

**Tip:** Run in a terminal with scrollback so you can review earlier steps!

## What Gets Saved

After training, you'll have:
- `checkpoints/best_model.pkl` - Best model (highest validation accuracy)
- Complete training history (loss/accuracy per epoch)
- Optimizer state (for resuming training)

## Interpreting Results

### Good Signs ✅
- Loss decreasing over epochs
- Train and Val accuracy both increasing
- Val accuracy close to Train accuracy (< 10% gap)
- First/Last operations have high accuracy (they're easiest)

### Warning Signs ⚠️
- Loss not decreasing after 5+ epochs
  → Try higher learning rate or check data
- Train accuracy >> Val accuracy (> 20% gap)
  → Overfitting! Stop training earlier or use more data
- All operations below 40% accuracy
  → Model not learning, check for bugs

### What to Expect
- **All Operations**: With medium+ dataset, expect 97-100% accuracy on all operations
  - Earlier assumption that Max/Min are "harder" was incorrect
  - Performance differences are mainly due to random initialization, not inherent difficulty
  - With proper training, all operations learn very well

## Visualization Tips

### Text-Based Attention Heatmap
```
High values (0.5+) = Strong attention
Low values (< 0.2) = Weak attention

For "First(2, 5, 8)":
- Should attend strongly to position of '2'

For "Max(2, 5, 8)":
- Should attend to all numbers to compare
```

### Pattern Types
- **Focused**: Low entropy, attends to specific positions
  - Expected for First/Second/Last
- **Distributed**: High entropy, attends everywhere
  - Expected for Max/Min
- **Mixed**: Medium entropy, partially focused
  - May indicate learning in progress

## Troubleshooting

### Error: "Out of memory"
```bash
# Use smaller batch size
python3 train_step_by_step.py --size large --batch-size 16
```

### Error: "Module not found"
```bash
# Make sure you're in the project directory
cd /home/grisha/dev/llmlearn
python3 train_step_by_step.py --size small
```

### Model not learning (accuracy stuck at ~10%)
```bash
# Try higher learning rate
python3 train_step_by_step.py --size small --lr 0.01

# Or more epochs
python3 train_step_by_step.py --size small --epochs 20
```

### Training too slow
```bash
# Use smaller dataset or larger batches
python3 train_step_by_step.py --size small --batch-size 128
```

## Next Steps After Training

1. **Save your results**: Take screenshots of the final summary

2. **Try different sizes**: Compare small vs medium vs large

3. **Experiment with hyperparameters**:
   - Learning rate: 0.0001, 0.001, 0.01
   - Batch size: 16, 32, 64, 128
   - Epochs: 5, 10, 20, 50

4. **Analyze patterns**:
   - Which operations are hardest?
   - Does attention make sense?
   - How does accuracy improve with data size?

5. **Plot training curves** (if matplotlib installed):
   ```python
   from evaluation import plot_training_curves
   # Use saved checkpoint
   ```

## Quick Reference

| Dataset | Examples | Epochs | Time      | Expected Acc |
|---------|----------|--------|-----------|--------------|
| Small   | 1,000    | 10     | 2-3 min   | 55-65%       |
| Medium  | 10,000   | 20     | ~6 min    | 93-99%       |
| Large   | 100,000  | 50     | 2-3 hrs   | 95%+         |

---

**Ready to train your transformer!** 🚀

Start with the small dataset to see it work, then scale up!
