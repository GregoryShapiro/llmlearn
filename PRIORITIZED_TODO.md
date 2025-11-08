# Prioritized TODO List
## LLMLearn From-Scratch Transformer

**Generated:** 2025-11-08
**Current Project Status:** 🟢 Fully Functional
**Technical Debt Level:** 🟢 Very Low

---

## Priority Legend

- 🔴 **CRITICAL** - Blocking issues, must fix immediately
- 🟠 **HIGH** - Important improvements, should do soon
- 🟡 **MEDIUM** - Nice to have, do when time permits
- 🟢 **LOW** - Future enhancements, optional
- 💡 **IDEA** - Interesting extensions, research needed

---

## Current Status: No Critical Issues! 🎉

The project is **production-ready for educational use** with no blocking issues.

---

## 🟠 HIGH Priority (Recommended Soon)

### 1. Enable Real Attention Visualization
**Status:** Framework ready, needs 1-line fix
**Effort:** 5 minutes
**Impact:** HIGH - Currently uses dummy data for attention visualization

**Location:** `src/layers/attention.py:346`

**Change needed:**
```python
# In MultiHeadAttention.forward(), after line 458, add:
self.last_attention_weights = attention_weights
```

**Files affected:**
- `src/layers/attention.py` - Add the line
- `src/visualization.py` - Already ready to use it

**Testing:**
```bash
# After fix, test with:
python3 -c "
import sys; sys.path.insert(0, 'src')
from visualization import extract_attention_weights
# Should extract real weights instead of dummy data
"
```

**Benefits:**
- See actual attention patterns from trained models
- Understand what the model learned
- Educational insight into attention mechanism

---

### 2. Add Type Hints to Core Files
**Status:** 45% complete (8/18 files have type hints)
**Effort:** 2-3 hours
**Impact:** MEDIUM - Better IDE support and documentation

**Files needing type hints:**
1. `src/transformer.py` (763 lines)
2. `src/loss.py` (381 lines)
3. `src/optimizer.py` (482 lines)
4. `src/train_utils.py` (312 lines)
5. `src/evaluation.py` (688 lines)
6. `src/visualization.py` (564 lines)

**Already complete:**
- ✅ `src/vocabluary.py`
- ✅ `src/data_generatpr.py`
- ✅ `src/data_utils.py`
- ✅ All files in `src/layers/`

**Example changes:**
```python
# Before:
def forward(self, x, mask=None):
    ...

# After:
def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    ...
```

**Benefits:**
- Better IDE autocomplete
- Catch type errors early
- Self-documenting code
- Easier for contributors

**Testing:**
```bash
# After changes:
pip install mypy
mypy src/ --ignore-missing-imports
```

---

## 🟡 MEDIUM Priority (Nice to Have)

### 3. Implement Dropout
**Status:** Mentioned in docstrings but not implemented
**Effort:** 2-3 hours
**Impact:** MEDIUM - Better generalization for larger models

**Locations needing dropout:**
- `src/layers/` - Create `dropout.py`
- `src/transformer.py` - Add to TransformerBlock
- `src/transformer_decoder.py` - Add to decoder blocks

**Implementation:**
```python
class Dropout:
    """
    Dropout layer for regularization.

    During training: Randomly zero out elements with probability p
    During inference: Keep all elements (no dropout)
    """
    def __init__(self, p=0.1):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        self.mask = np.random.binomial(1, 1-self.p, x.shape) / (1-self.p)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask if self.training else grad_output
```

**Integration points:**
- After attention in TransformerBlock
- After FFN in TransformerBlock
- Typical dropout rate: 0.1

**Benefits:**
- Prevents overfitting on larger datasets
- Standard transformer component
- More realistic implementation

**Testing:**
- Add tests in `tests/test_layers.py`
- Verify mask is used in training mode
- Verify no dropout in eval mode

---

### 4. Add Learning Rate Scheduling
**Status:** Not implemented
**Effort:** 1-2 hours
**Impact:** MEDIUM - Better convergence

**Common schedulers to implement:**

**A) Step Decay:**
```python
class StepLR:
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma
```

**B) Warmup + Cosine Annealing:**
```python
def get_lr(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

**C) ReduceLROnPlateau:**
```python
class ReduceLROnPlateau:
    def __init__(self, optimizer, patience=5, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.wait = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.optimizer.learning_rate *= self.factor
                self.wait = 0
```

**Recommended:** Start with ReduceLROnPlateau (easiest to use)

**Benefits:**
- Better final accuracy
- Faster convergence
- Standard training practice

**Integration:**
- Add `schedulers.py` to `src/`
- Update training scripts to use scheduler
- Add to `train_step_by_step.py` first

---

### 5. Create Checkpoints Directory on Install
**Status:** Gitignored but not auto-created
**Effort:** 5 minutes
**Impact:** LOW - Scripts already handle this, but cleaner UX

**Option A: Add to training scripts**
```python
import os
checkpoint_dir = 'checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)
```

**Option B: Add setup script**
```bash
# setup.sh
#!/bin/bash
mkdir -p checkpoints
echo "Setup complete!"
```

**Option C: Add to README**
```bash
# Installation
git clone <repo>
cd llmlearn
mkdir checkpoints  # Add this line
pip install numpy matplotlib
```

**Recommendation:** Option A (already done in most scripts)

---

### 6. Add More Training Examples in Documentation
**Status:** Good but could expand
**Effort:** 1 hour
**Impact:** MEDIUM - Helps new users

**Additions needed:**
1. **Common issues troubleshooting:**
   - "Loss not decreasing" → check learning rate
   - "Accuracy stuck at 20%" → check data is balanced
   - "NaN loss" → learning rate too high

2. **Hyperparameter tuning guide:**
   - Effect of `embed_dim` (32 vs 64 vs 128)
   - Effect of `num_heads` (2 vs 4 vs 8)
   - Effect of `num_layers` (1 vs 2 vs 4)
   - Learning rate sweep results

3. **Performance benchmarks:**
   - Training time vs dataset size
   - Accuracy vs epochs
   - Memory usage

**File:** Add new `TROUBLESHOOTING.md`

---

## 🟢 LOW Priority (Future Enhancements)

### 7. Add PyTorch Reference Implementation
**Status:** Not started
**Effort:** 4-6 hours
**Impact:** LOW - Educational comparison

**Purpose:**
- Verify numerical correctness
- Compare speed (NumPy vs PyTorch)
- Teaching tool (side-by-side comparison)

**Structure:**
```
src/
├── pytorch_reference/
│   ├── transformer_pytorch.py
│   ├── compare_outputs.py
│   └── README.md
```

**Comparison points:**
- Forward pass outputs (should match within 1e-5)
- Gradient values (should match within 1e-5)
- Training curves (should be similar)
- Speed (PyTorch should be 5-10x faster on CPU, 100x on GPU)

**Benefits:**
- Validate implementation correctness
- Educational value (see both approaches)
- Debugging tool

---

### 8. Implement Beam Search for Decoder
**Status:** Only greedy decoding implemented
**Effort:** 3-4 hours
**Impact:** LOW - More diverse generation

**Current:**
```python
# Greedy decoding (always pick most likely token)
next_token = np.argmax(logits, axis=-1)
```

**Beam search:**
```python
def beam_search(model, start_tokens, beam_width=5, max_len=20):
    """
    Generate sequences using beam search.

    Maintains top-k hypotheses at each step.
    """
    beams = [(start_tokens, 0.0)]  # (sequence, score)

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model.forward(seq)
            probs = softmax(logits[-1])

            # Get top-k tokens
            top_k_idx = np.argsort(probs)[-beam_width:]
            for idx in top_k_idx:
                new_seq = np.append(seq, idx)
                new_score = score + np.log(probs[idx])
                candidates.append((new_seq, new_score))

        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Check for EOS token
        # ...

    return beams[0][0]  # Return best sequence
```

**Benefits:**
- More diverse outputs
- Better quality for open-ended generation
- Standard decoding method

**Use case:**
- Decoder/autoregressive models only
- Not needed for encoder (classification)

---

### 9. Add Nested Operations Support
**Status:** Not implemented
**Effort:** 6-8 hours
**Impact:** LOW - Significantly more complex

**Current:**
```
Max(3, 5, 7) → 7
```

**Nested:**
```
Max(First(2, 8), Second(5, 1)) → Max(2, 5) → 5
```

**Challenges:**
- Parsing nested expressions
- Variable sequence length
- Recursive evaluation
- More complex tokenization

**Implementation steps:**
1. Extend vocabulary with parentheses
2. Build expression tree parser
3. Evaluate tree recursively
4. Generate training data with nesting
5. Update model architecture (may need larger context)

**Recommendation:** New project scope, not a minor addition

---

### 10. Add Gradient Flow Visualization
**Status:** Not implemented
**Effort:** 3-4 hours
**Impact:** LOW - Educational value

**Visualizations to add:**
```python
def plot_gradient_flow(model):
    """
    Plot gradient magnitudes per layer.

    Helps identify vanishing/exploding gradients.
    """
    layers = []
    grads = []

    for name, (param, grad) in model.get_parameters():
        if grad is not None:
            layers.append(name)
            grads.append(np.abs(grad).mean())

    plt.bar(layers, grads)
    plt.xticks(rotation=45)
    plt.ylabel('Mean Gradient Magnitude')
    plt.title('Gradient Flow')
    plt.show()
```

**Additional visualizations:**
- Parameter histograms
- Activation distributions
- Weight evolution over training
- Loss landscape (2D slice)

**Benefits:**
- Debug training issues
- Educational tool
- Understand network behavior

---

### 11. Add t-SNE Embedding Visualization
**Status:** Not implemented
**Effort:** 2 hours
**Impact:** LOW - Nice visualization

**Implementation:**
```python
def visualize_embeddings(model, vocabulary):
    """
    Visualize token embeddings in 2D using t-SNE.

    Shows which tokens the model thinks are similar.
    """
    from sklearn.manifold import TSNE

    embeddings = model.embedding.embedding_matrix  # (vocab_size, embed_dim)

    # Reduce to 2D
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    for i, token in enumerate(vocabulary):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(token, (x, y))

    plt.title('Token Embeddings (t-SNE)')
    plt.show()
```

**Expected results:**
- Digits 0-9 should cluster together
- Operations should cluster together
- Similar operations (Max/Min) should be close

**Note:** Requires scikit-learn dependency

---

### 12. Add Explicit Gradient Clipping
**Status:** Implicitly handled by optimizer
**Effort:** 30 minutes
**Impact:** LOW - Already stable

**Current:** Adam naturally limits gradients via adaptive learning rates

**Explicit clipping:**
```python
def clip_gradients(parameters, max_norm=1.0):
    """
    Clip gradients by global norm.

    Prevents exploding gradients.
    """
    total_norm = 0
    for param, grad in parameters:
        if grad is not None:
            total_norm += np.sum(grad ** 2)

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for param, grad in parameters:
            if grad is not None:
                grad *= clip_coef
```

**When to use:**
- Training unstable (loss spikes)
- Very deep networks
- Recurrent models

**For this project:** Not needed (Adam handles it)

---

## 💡 IDEA Priority (Research/Exploration)

### 13. Compare Different Architectures
**Effort:** Variable
**Impact:** Educational

**Experiments to try:**
1. **Positional encoding variants:**
   - Learned positional embeddings
   - Rotary positional embeddings (RoPE)
   - ALiBi (attention with linear biases)

2. **Normalization variants:**
   - Pre-LN vs Post-LN
   - RMSNorm (simpler than LayerNorm)
   - No normalization (does it work?)

3. **Attention variants:**
   - Sparse attention patterns
   - Local attention (windowed)
   - Relative position attention

4. **Activation variants:**
   - GELU vs ReLU
   - SwiGLU (used in LLaMA)
   - Mish

**Document results:** Create `ARCHITECTURE_EXPERIMENTS.md`

---

### 14. Implement Label Smoothing
**Effort:** 1 hour
**Impact:** Potentially better generalization

**Current loss:**
```python
# Hard target: [0, 0, 1, 0, 0] (one-hot)
loss = -log(p[true_class])
```

**Label smoothing:**
```python
# Soft target: [0.05, 0.05, 0.8, 0.05, 0.05]
def smooth_labels(targets, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)

    one_hot = np.eye(num_classes)[targets]
    smooth_targets = one_hot * confidence + smooth_value

    return smooth_targets
```

**Benefits:**
- Prevents overconfident predictions
- Better calibrated probabilities
- Slight accuracy improvement

**Use case:** When model accuracy is already high (>95%)

---

### 15. Add Mixed Precision Training (If Supported)
**Effort:** 3-4 hours
**Impact:** Faster training, lower memory

**Challenge:** NumPy doesn't support mixed precision easily

**Partial solution:**
```python
# Store weights in float32, compute in float16
class MixedPrecisionLinear:
    def __init__(self, ...):
        self.weight = np.random.randn(...).astype(np.float32)

    def forward(self, x):
        # Convert to float16 for computation
        x_fp16 = x.astype(np.float16)
        w_fp16 = self.weight.astype(np.float16)
        output = np.matmul(x_fp16, w_fp16)
        return output.astype(np.float32)
```

**Limitation:** NumPy float16 has limited precision, may cause issues

**Recommendation:** Not worth it for NumPy implementation

---

### 16. Create Interactive Web Demo
**Effort:** 6-8 hours
**Impact:** Great for demonstrations

**Tech stack:**
- Flask backend
- Simple HTML/CSS/JS frontend
- Model loaded in memory

**Features:**
```python
# app.py
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model once at startup
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    operation = request.json['operation']
    # Tokenize, predict, return result
    return jsonify({'answer': result})

if __name__ == '__main__':
    app.run(debug=True)
```

**Frontend:**
```html
<input type="text" id="operation" placeholder="Max(3, 7, 5)">
<button onclick="predict()">Predict</button>
<div id="result"></div>
```

**Deployment:** Could use Heroku free tier or GitHub Pages + serverless

---

### 17. Add Profiling Tools
**Effort:** 2 hours
**Impact:** Identify bottlenecks

**Implementation:**
```python
import cProfile
import pstats

# Profile training
profiler = cProfile.Profile()
profiler.enable()

# Train for 1 epoch
train_epoch(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Expected bottlenecks:**
- Matrix multiplication in attention
- Embedding lookup
- Softmax computation

**Optimization opportunities:**
- Use BLAS-optimized NumPy
- Vectorize loops
- Cache repeated computations

**Note:** For educational project, optimization not critical

---

### 18. Create Jupyter Notebook Tutorial
**Effort:** 4-6 hours
**Impact:** Better onboarding

**Structure:**
```
notebooks/
├── 01_introduction.ipynb
├── 02_data_pipeline.ipynb
├── 03_attention_mechanism.ipynb
├── 04_transformer_blocks.ipynb
├── 05_training.ipynb
└── 06_evaluation.ipynb
```

**Each notebook:**
- Explanation of concepts
- Code snippets with output
- Visualizations
- Exercises for learners

**Benefits:**
- Interactive learning
- Visual explanations
- Self-paced tutorials

---

## 📊 Summary Statistics

### Current Completion
- **Total tasks identified:** 18
- **🔴 Critical:** 0 (none!)
- **🟠 High priority:** 2
- **🟡 Medium priority:** 5
- **🟢 Low priority:** 6
- **💡 Ideas:** 5

### Estimated Effort
- **High priority:** 2-3 hours total
- **Medium priority:** 7-11 hours total
- **Low priority:** 25-35 hours total
- **Ideas:** 20-40 hours total

### Recommended Next Steps

**Week 1:**
1. ✅ Fix attention visualization (5 min)
2. ✅ Add type hints (2-3 hours)

**Week 2-3:**
3. Implement dropout (2-3 hours)
4. Add learning rate scheduling (1-2 hours)

**Month 2-3 (optional):**
- Pick 2-3 items from LOW or IDEA priority
- Document experiments
- Share findings

---

## 🎯 Quick Wins (Do First)

These are high-impact, low-effort improvements:

1. **Fix attention visualization** (5 min) ⚡
2. **Add StepLR scheduler** (30 min) ⚡
3. **Create TROUBLESHOOTING.md** (1 hour) ⚡
4. **Add type hints to loss.py** (20 min) ⚡
5. **Add gradient clipping example** (15 min) ⚡

---

## 📝 Notes

### What NOT to Do

❌ **Don't add GPU support** - Beyond scope, use PyTorch instead
❌ **Don't optimize for speed** - Educational clarity > performance
❌ **Don't add complex features** - Keep it understandable
❌ **Don't change core architecture** - It works well as-is

### Maintenance Philosophy

This project prioritizes:
1. **Educational value** over performance
2. **Code clarity** over optimization
3. **Completeness** over bleeding-edge features
4. **Documentation** over code quantity

Keep additions aligned with these principles.

---

## 📅 Recommended Timeline

### Immediate (This Week)
- Fix attention visualization
- Add type hints to 2-3 files

### Short Term (This Month)
- Complete type hints
- Add dropout
- Add learning rate scheduler

### Medium Term (3 Months)
- Pick 2-3 LOW priority items
- Create troubleshooting guide
- Document experiments

### Long Term (6+ Months)
- Jupyter notebook tutorial
- Web demo
- Advanced architecture experiments

---

## 🏁 Conclusion

**The project is in excellent shape!**

No critical issues exist. The HIGH priority items are polish, not fixes. All MEDIUM and LOW priority items are **optional enhancements** that would be nice but aren't necessary.

**Recommendation:**
1. Do the 2 HIGH priority items (3 hours total)
2. Consider project complete for v1.0
3. Explore MEDIUM/LOW items only if extending the project

**Current State:** ⭐⭐⭐⭐⭐ (5/5) - Production-ready for educational use

---

**Last Updated:** 2025-11-08
**Next Review:** After implementing HIGH priority items
