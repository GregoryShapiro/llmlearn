# How Real LLMs Work (vs Our Model)

## TL;DR - The Core Difference

**Real LLMs (GPT, Llama):**
```python
Input:  "Max(23, 45, 89) ="
Output: Generate autoregressively → "8" → "9" → stop
```

**Our Model:**
```python
Input:  "Max(23, 45, 89)"
Output: Predict single token → "8" (only first digit!)
```

---

## The 3 Key Differences

### 1. ❌ Causal Masking (We Don't Have This)

**Real LLMs use causal attention:**
```
Attention mask:
  Position 0 can see: [0]           ← Only past
  Position 1 can see: [0, 1]        ← Only past
  Position 2 can see: [0, 1, 2]     ← Only past
```

**Our model uses full attention:**
```
All positions can see everything (including future):
  Position 0 can see: [0, 1, 2, 3, ...]  ← Can see future!
  Position 1 can see: [0, 1, 2, 3, ...]
```

**Why this matters:**
- Real LLMs can't "cheat" by seeing the answer
- Our model sees the whole sequence at once (encoder-style)
- This is why we can't generate sequences autoregressively

**How to fix:**
```python
# In scaled_dot_product_attention():
# Create causal mask
seq_len = Q.shape[1]
causal_mask = np.tril(np.ones((seq_len, seq_len)))  # Lower triangular
#  [[1, 0, 0],
#   [1, 1, 0],
#   [1, 1, 1]]

scores = scores + (1 - causal_mask) * -1e9
```

---

### 2. ❌ Autoregressive Training (We Only Train on First Digit)

**Real LLMs train on EVERY position:**

```python
# Training example 1:
Input:  "Max(23, 45, 89) ="
Target: "8"

# Training example 2:
Input:  "Max(23, 45, 89) = 8"
Target: "9"

# Training example 3:
Input:  "Max(23, 45, 89) = 8 9"
Target: "[EOS]"
```

**Our model trains on FIRST DIGIT ONLY:**

```python
# We only create ONE training example:
Input:  "Max(23, 45, 89)"
Target: "8"  # First digit only!

# We never teach it to predict "9" or when to stop
```

**Our current code:**
```python
# train_double_digits.py, line 147:
train_targets = train_targets[:, 0]  # ← THIS IS THE PROBLEM!
```

**How real LLMs do it:**
```python
# For each output position, create a training example
for i in range(len(answer_tokens)):
    input_seq = question + answer_tokens[:i]
    target = answer_tokens[i]
    train_on(input_seq, target)
```

---

### 3. ❌ Sequence-to-Sequence Output (We Only Predict One Token)

**Real LLMs generate until stop token:**

```python
def generate(model, prompt):
    tokens = tokenize(prompt)

    while True:
        next_token = model.predict(tokens)  # Predict next

        if next_token == EOS_TOKEN:
            break

        tokens.append(next_token)  # Add to sequence
        # Feed back for next prediction ← AUTOREGRESSIVE

    return tokens
```

**Our model predicts once:**

```python
def predict(model, input):
    output = model.forward(input)  # Shape: (batch, vocab_size)
    prediction = argmax(output)    # Single token
    return prediction              # Done!
```

**Why ours fails for multi-digit:**
```
Input:  Max(23, 45, 89)
Model predicts: 8
Model stops. ← No mechanism to continue!
```

---

## Complete Comparison Table

| Feature | Real LLMs (GPT/Llama) | Our Model |
|---------|----------------------|-----------|
| **Architecture** | Decoder-only | Encoder-only |
| **Attention Type** | Causal (masked) | Full (bidirectional) |
| **Training** | Next token prediction | Single token classification |
| **Output Length** | Variable (until EOS) | Fixed (1 token) |
| **Generation** | Autoregressive loop | Single forward pass |
| **Can see future?** | No ✓ | Yes ✗ |
| **Multi-turn?** | Yes ✓ | No ✗ |
| **Multi-digit output?** | Yes ✓ | No ✗ |

---

## How GPT/Llama Actually Work

### Training Phase:

```python
# Input text: "The answer is 42"
# Tokens: ["The", "answer", "is", "42"]

# Create training examples:
Input: ["The"]                  → Target: "answer"
Input: ["The", "answer"]        → Target: "is"
Input: ["The", "answer", "is"]  → Target: "42"

# Train with causal masking (can't see future)
```

### Inference Phase:

```python
# User prompt: "What is 2+2?"
tokens = ["What", "is", "2", "+", "2", "?"]

# Generate response autoregressively:
Step 1: model.predict(tokens) → "The"
        tokens = [..., "The"]

Step 2: model.predict(tokens) → "answer"
        tokens = [..., "The", "answer"]

Step 3: model.predict(tokens) → "is"
        tokens = [..., "The", "answer", "is"]

Step 4: model.predict(tokens) → "4"
        tokens = [..., "is", "4"]

Step 5: model.predict(tokens) → "[EOS]"
        STOP!

# Final: "The answer is 4"
```

---

## Why Our Model Architecture is Different

### We Built an Encoder (Like BERT)

**Good for:**
- Classification (sentiment analysis)
- Understanding (question answering with predefined answers)
- Embedding (sentence similarity)

**Bad for:**
- Text generation
- Multi-token outputs
- Conversational AI

### LLMs Are Decoders (Like GPT)

**Good for:**
- Text generation
- Multi-token answers
- Conversational AI
- Creative writing

**Bad for:**
- Requires more data
- Slower (autoregressive = sequential)
- Harder to train

---

## How to Convert Our Model to Work Like GPT

### Step 1: Add Causal Masking

```python
# In transformer.py forward():
def forward(self, x):
    # ...
    seq_len = x.shape[1]
    causal_mask = np.tril(np.ones((seq_len, seq_len)))

    # Pass mask to attention
    x, attn = self.attention(x, mask=causal_mask)
```

### Step 2: Change Training Data Format

```python
# OLD (classification):
Input:  Max(23, 45, 89)
Target: 8  # First digit only

# NEW (autoregressive):
# Concatenate input + answer
Full_sequence: Max ( 2 3 , 4 5 , 8 9 ) = 8 9 [EOS]

# Train to predict NEXT token at every position:
Position 0:  Input=""              → Predict="Max"
Position 1:  Input="Max"           → Predict="("
...
Position 11: Input="Max(...)="     → Predict="8"
Position 12: Input="Max(...)=8"    → Predict="9"
Position 13: Input="Max(...)=89"   → Predict="[EOS]"
```

### Step 3: Implement Autoregressive Generation

```python
def generate_answer(model, question_tokens, max_len=5):
    """Generate answer autoregressively like GPT."""

    # Start with question + "=" token
    sequence = question_tokens + [VOCAB['=']]

    for _ in range(max_len):
        # Predict next token
        logits = model.forward(sequence)
        next_token = np.argmax(logits[-1])  # Last position

        # Stop if EOS
        if next_token == VOCAB['[EOS]']:
            break

        # Add to sequence
        sequence.append(next_token)

    # Extract answer (everything after "=")
    answer_start = sequence.index(VOCAB['=']) + 1
    return sequence[answer_start:]
```

### Step 4: Modify Model Output

```python
# OLD: Output single prediction
def forward(self, x):
    # ...
    return output[0]  # Only first position

# NEW: Output prediction for EACH position
def forward(self, x):
    # ...
    return output  # All positions (batch, seq_len, vocab_size)
```

---

## Why We Didn't Build It This Way

### Educational Reasons:
1. **Simpler**: Classification is easier to understand than generation
2. **Faster**: Single forward pass vs autoregressive loop
3. **Less data**: Classification needs less training data
4. **Clearer evaluation**: Accuracy is straightforward

### Technical Reasons:
1. **Smaller dataset**: 10K examples enough for classification
2. **Faster training**: No need to expand examples by sequence length
3. **Easier debugging**: Simpler data pipeline

---

## Real LLM Training Scale

### Our Model:
- **Dataset**: 10,000 examples
- **Parameters**: ~100K
- **Training time**: 5 minutes
- **Hardware**: CPU

### GPT-3:
- **Dataset**: 300B tokens (~500GB of text)
- **Parameters**: 175B
- **Training time**: Weeks
- **Hardware**: Thousands of GPUs
- **Cost**: ~$4.6 million

### Llama 3.1 (405B):
- **Dataset**: 15T tokens
- **Parameters**: 405B
- **Training time**: Months
- **Hardware**: 16,000 GPUs
- **Cost**: $100M+

---

## Summary: What You Need for Multi-Digit Output

### Option 1: Proper Autoregressive Model (Like GPT) ✓ Correct
```
1. Add causal masking
2. Change training to next-token prediction
3. Implement autoregressive generation loop
4. Train on full sequences (not just first digit)
```

### Option 2: Sequence-to-Sequence (Like T5) ✓ Also works
```
1. Keep encoder
2. Add decoder with causal masking
3. Train encoder-decoder jointly
4. Generate with decoder
```

### Option 3: Hack Current Model ✗ Doesn't work
```
Try to generate autoregressively without proper training
→ Model wasn't trained for this
→ Produces gibberish (as we saw: "99", "33", etc.)
```

---

## Next Steps to Fix Our Model

1. **Quick fix (for learning):**
   - Keep single-digit model
   - Document limitations clearly

2. **Proper fix (for multi-digit):**
   - Implement causal masking
   - Create autoregressive training data
   - Train new model with next-token prediction
   - Implement generation loop

3. **Best practice (for real use):**
   - Use existing LLM frameworks (PyTorch, transformers library)
   - Don't reinvent the wheel for production
   - Our NumPy implementation is for education only

---

**Key Takeaway:**

Our model is an **encoder** (like BERT) that's great for classification.

Real LLMs are **decoders** (like GPT) that generate text autoregressively.

To do multi-digit output properly, we need decoder architecture with causal masking and autoregressive training. The current model can't be "fixed" with just better inference - it needs to be retrained differently.

---

**Created**: 2025-10-21
**Purpose**: Educational explanation of LLM architectures
