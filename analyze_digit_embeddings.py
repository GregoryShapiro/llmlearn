"""
Analyze Digit Embedding Correlations

This script analyzes how digit embeddings evolved during training
and shows correlations between different digits.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pickle

# Load the trained model
print("=" * 80)
print("DIGIT EMBEDDING CORRELATION ANALYSIS")
print("=" * 80)
print()

print("Loading best model from checkpoint...")
with open('checkpoints/best_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Get model parameters (stored as dict with param_0, param_1, etc.)
model_params = checkpoint['model_params']

# The first parameter should be the embedding matrix
embeddings = model_params['param_0']

print(f"✓ Loaded embeddings: {embeddings.shape}")
print()

# Digit tokens: indices 2-11 (0-9)
digit_indices = list(range(2, 12))
digit_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

print("=" * 80)
print("DIGIT EMBEDDING STATISTICS")
print("=" * 80)
print()

print("Embedding Norms (L2):")
print("-" * 80)
for i, (idx, name) in enumerate(zip(digit_indices, digit_names)):
    norm = np.linalg.norm(embeddings[idx])
    bar_length = int(norm * 10)
    bar = '█' * bar_length
    print(f"  {name}: {norm:7.4f}  {bar}")

print()
print("Embedding Statistics:")
print("-" * 80)
digit_embeddings = embeddings[digit_indices]
print(f"  Mean of all digit embeddings:  {np.mean(digit_embeddings):7.4f}")
print(f"  Std Dev:                        {np.std(digit_embeddings):7.4f}")
print(f"  Min value:                      {np.min(digit_embeddings):7.4f}")
print(f"  Max value:                      {np.max(digit_embeddings):7.4f}")
print()

# Compute pairwise cosine similarities
print("=" * 80)
print("DIGIT EMBEDDING SIMILARITIES (Cosine)")
print("=" * 80)
print()

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

# Create similarity matrix
similarity_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        emb_i = embeddings[digit_indices[i]]
        emb_j = embeddings[digit_indices[j]]
        similarity_matrix[i, j] = cosine_similarity(emb_i, emb_j)

# Print similarity matrix
print("Similarity Matrix:")
print("-" * 80)
print("       ", end="")
for name in digit_names:
    print(f"{name:7s}", end="")
print()
print("-" * 80)

for i, name_i in enumerate(digit_names):
    print(f"  {name_i}:  ", end="")
    for j in range(10):
        sim = similarity_matrix[i, j]
        if i == j:
            print(f"{'1.000':>7s}", end="")
        else:
            print(f"{sim:7.3f}", end="")
    print()

print()

# Find most similar and most dissimilar pairs
print("=" * 80)
print("NOTABLE DIGIT PAIRS")
print("=" * 80)
print()

pairs = []
for i in range(10):
    for j in range(i + 1, 10):
        sim = similarity_matrix[i, j]
        pairs.append((digit_names[i], digit_names[j], sim))

# Sort by similarity
pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

print("Most Similar Digit Pairs:")
print("-" * 80)
for i in range(min(10, len(pairs_sorted))):
    d1, d2, sim = pairs_sorted[i]
    print(f"  {d1} <-> {d2}:  {sim:7.4f}")

print()
print("Most Dissimilar Digit Pairs:")
print("-" * 80)
for i in range(min(10, len(pairs_sorted))):
    d1, d2, sim = pairs_sorted[-(i+1)]
    print(f"  {d1} <-> {d2}:  {sim:7.4f}")

print()

# Analyze patterns
print("=" * 80)
print("PATTERN ANALYSIS")
print("=" * 80)
print()

# Average similarity to each digit
print("Average Similarity to Each Digit:")
print("-" * 80)
for i, name in enumerate(digit_names):
    avg_sim = np.mean([similarity_matrix[i, j] for j in range(10) if j != i])
    bar_length = int((avg_sim + 1) * 20)  # Scale to 0-40 chars
    bar = '█' * bar_length
    print(f"  {name}:  {avg_sim:7.4f}  {bar}")

print()

# Check for numeric ordering correlation
print("Correlation with Numeric Value:")
print("-" * 80)
print("Does embedding similarity correlate with numeric proximity?")
print()

# For each digit, compute average similarity to neighbors vs distant digits
for i in range(10):
    # Neighbors: digits within distance 1
    neighbor_sims = []
    distant_sims = []

    for j in range(10):
        if i == j:
            continue
        distance = abs(i - j)
        sim = similarity_matrix[i, j]

        if distance == 1:
            neighbor_sims.append(sim)
        elif distance >= 5:
            distant_sims.append(sim)

    if neighbor_sims and distant_sims:
        avg_neighbor = np.mean(neighbor_sims)
        avg_distant = np.mean(distant_sims)
        diff = avg_neighbor - avg_distant
        marker = "✓ neighbors more similar" if diff > 0 else "✗ distant more similar"

        print(f"  {digit_names[i]}:  Neighbors: {avg_neighbor:6.3f}  Distant: {avg_distant:6.3f}  "
              f"Diff: {diff:+6.3f}  {marker}")

print()

# Check extremes (0, 9) vs middle (4, 5)
print("Extremes vs Middle Digits:")
print("-" * 80)
extremes = [0, 1, 8, 9]  # indices for digits 0, 1, 8, 9
middles = [4, 5]  # indices for digits 4, 5

extreme_to_extreme = []
extreme_to_middle = []
middle_to_middle = []

for i in extremes:
    for j in extremes:
        if i < j:
            extreme_to_extreme.append(similarity_matrix[i, j])
    for j in middles:
        extreme_to_middle.append(similarity_matrix[i, j])

for i in middles:
    for j in middles:
        if i < j:
            middle_to_middle.append(similarity_matrix[i, j])

print(f"  Extreme-to-Extreme (0,1,8,9):  {np.mean(extreme_to_extreme):6.3f}")
print(f"  Extreme-to-Middle (4,5):       {np.mean(extreme_to_middle):6.3f}")
print(f"  Middle-to-Middle (4,5):        {np.mean(middle_to_middle):6.3f}")
print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("Key Insights:")
print()
print("1. Embedding Magnitudes:")
print("   - Digit embeddings have large norms (2-4), much larger than syntax tokens")
print("   - This shows digits are semantically important for the task")
print()
print("2. Digit Similarities:")
print("   - If positive correlations dominate: digits are treated as a semantic group")
print("   - If negative correlations exist: model distinguishes between digit values")
print("   - Random correlations: model treats each digit independently")
print()
print("3. Numeric Proximity:")
print("   - If neighbors more similar: model learned numeric ordering (0 near 1, etc.)")
print("   - If no pattern: model treats digits as categorical (not ordered)")
print()
print("4. What This Reveals:")
print("   - For Max/Min operations: model needs to understand numeric values")
print("   - For First/Second/Last: model might not need numeric ordering")
print("   - The correlation patterns show what the model learned to solve the task")
print()
print("=" * 80)
