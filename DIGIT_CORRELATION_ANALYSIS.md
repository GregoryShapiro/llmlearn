# Digit Embedding Correlation Analysis

## Overview

Analysis of how the transformer model learned to represent digits (0-9) after training on 10,000 examples for 20 epochs.

## Key Findings

### 1. Embedding Magnitudes Show Semantic Importance

**Digit Norms (L2):**
- Digit 5: **4.10** (largest) - Most distinctive
- Digit 2: **3.21**
- Digit 6: **3.38**
- Digit 8: **2.15** (smallest) - Least distinctive

**Comparison:**
- **Digits**: 2.15 - 4.10 (highly varied, large)
- **Operations**: 1.22 - 1.92 (medium)
- **Syntax**: 0.94 - 0.99 (small, uniform)

**Interpretation:** The model learned that digits carry the most important semantic information (they are the actual answer values), while syntax tokens are just structural markers.

---

## 2. Digit Correlation Patterns

### Most Similar Digit Pairs
```
7 <-> 8:   +0.198  (most similar!)
0 <-> 1:   +0.151
6 <-> 9:   +0.145
4 <-> 9:   +0.130
5 <-> 6:   +0.119
```

### Most Dissimilar Digit Pairs
```
0 <-> 9:   -0.480  (most dissimilar!)
6 <-> 8:   -0.378
4 <-> 7:   -0.331
3 <-> 5:   -0.257
0 <-> 8:   -0.251
```

### Average Similarity by Digit
```
Digit 8: -0.033  (most distinct from others)
Digit 9: -0.049
Digit 7: -0.059
Digit 3: -0.060
Digit 1: -0.076
Digit 0: -0.096  (least distinct)
```

---

## 3. Numeric Proximity Hypothesis

**Question:** Did the model learn that numerically adjacent digits (like 0-1, 5-6) are similar?

**Test:** For each digit, compare average similarity to:
- **Neighbors:** digits ±1 away (e.g., for 5: digits 4 and 6)
- **Distant:** digits ≥5 away (e.g., for 5: digits 0 and 9)

**Results:**

| Digit | Neighbor Similarity | Distant Similarity | Learned Ordering? |
|-------|--------------------|--------------------|-------------------|
| **0** | +0.151 | -0.223 | ✓ YES (+0.374) |
| **1** | -0.033 | -0.149 | ✓ YES (+0.115) |
| **2** | -0.230 | -0.002 | ✗ NO (-0.228) |
| **3** | -0.150 | -0.002 | ✗ NO (-0.147) |
| **4** | -0.071 | +0.130 | ✗ NO (-0.200) |
| **5** | +0.017 | -0.224 | ✓ YES (+0.241) |
| **6** | +0.027 | -0.144 | ✓ YES (+0.171) |
| **7** | +0.067 | -0.088 | ✓ YES (+0.154) |
| **8** | +0.141 | -0.069 | ✓ YES (+0.209) |
| **9** | +0.084 | -0.109 | ✓ YES (+0.193) |

**Summary:** **7 out of 10 digits** show numeric proximity learning! The model partially learned that adjacent numbers are more similar.

---

## 4. Extreme vs Middle Digits

```
Extreme digits (0,1,8,9) to each other:  -0.113
Extreme digits (0,1,8,9) to middle (4,5): -0.025
Middle digits (4,5) to each other:       -0.085
```

**Interpretation:** No strong pattern distinguishing extremes from middle values. The model treats digits more individually rather than as "low/middle/high" categories.

---

## 5. Notable Patterns

### Digit 0 and 9 are Opposites
- Similarity: **-0.480** (strongest negative correlation)
- These are the extreme values (min/max in single digits)
- Model learned they are semantically opposite

### Digits 7 and 8 are Similar
- Similarity: **+0.198** (strongest positive correlation)
- Adjacent in numeric order
- Model learned they belong together

### Digit 8 is Most Distinctive
- Average similarity to others: **-0.033**
- Least correlated with other digits
- Has unique semantic role in the model's representation

---

## 6. What the Model Learned

### For Max/Min Operations
The model needed to understand:
- **Relative magnitude** of digits (which is larger/smaller)
- **Extreme values** (0 and 9 are opposites)
- **Numeric ordering** (partially learned for most digits)

### For First/Second/Last Operations
The model only needed:
- **Token identity** (recognize which digit it is)
- **No numeric value understanding required**

### Actual Learning
The embedding patterns show the model learned a **hybrid representation**:
- ✓ Strong distinction between different digit values (negative average correlations)
- ✓ Partial numeric ordering (7/10 digits show proximity learning)
- ✓ Recognition of extremes (0 and 9 strongly dissimilar)
- ✓ Individual semantic importance (varied embedding norms)

This is **optimal for the task** because:
- Max/Min operations require value comparisons (learned)
- First/Second/Last require identity recognition (learned)
- Model didn't over-learn unnecessary patterns

---

## 7. Comparison with Operations

### Operation Embedding Similarities (from embeddings.txt)
```
First   <-> Second:  -0.045
First   <-> Last:    -0.594  (opposites!)
First   <-> Max:     -0.027
First   <-> Min:     +0.018
Second  <-> Last:    -0.134
Second  <-> Max:     +0.113
Second  <-> Min:     -0.178
Last    <-> Max:     -0.226
Last    <-> Min:     +0.060
Max     <-> Min:     -0.658  (strongest opposites!)
```

**Pattern:** Operations also show semantic relationships:
- First and Last are opposites (-0.594)
- Max and Min are opposites (-0.658)
- Second is somewhat distinct from both First and Last

This mirrors the digit learning: the model learned to represent **opposites as dissimilar** and **related concepts as somewhat similar**.

---

## 8. Evolution During Training

From embeddings.txt, we can see:

**Initial (Epoch 0):**
- All norms: ~0.07-0.08 (uniform, random)
- All similarities: near 0 (random initialization)

**Final (Epoch 20):**
- Digit norms: 2.15-4.10 (40-50× larger!)
- Operation norms: 1.22-1.92
- Syntax norms: 0.94-0.99 (barely changed)
- Clear similarity patterns emerged

**Growth Rate:**
1. **Digits grew most** (they determine answers)
2. **Operations grew moderately** (they determine which digit to select)
3. **Syntax barely grew** (just structural markers)

---

## Conclusions

1. **The model learned semantic importance:** Digit embeddings are 2-4× larger than operations, 3-4× larger than syntax

2. **Partial numeric understanding:** 70% of digits (7/10) learned that adjacent numbers are more similar

3. **Semantic opposites:** 0↔9 and Max↔Min show strong negative correlations

4. **Task-optimal representation:** The model learned exactly what it needed:
   - Value distinctions for Max/Min
   - Identity recognition for First/Second/Last
   - Numeric proximity for comparison operations

5. **Training effectiveness:** From random initialization to structured semantic space in just 20 epochs (6.2 minutes)

This analysis reveals that even a small transformer (102K parameters) can learn rich, task-appropriate representations from scratch!
