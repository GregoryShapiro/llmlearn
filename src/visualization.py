"""
Attention Visualization Module

This module provides tools to visualize and analyze attention patterns in the transformer.

Why Visualize Attention?
    Attention weights tell us what the model is "looking at" when making predictions.
    For our problem:
    - First(a, b, c) should attend to position of 'a'
    - Last(a, b, c) should attend to position of 'c'
    - Max/Min should attend to all numbers to compare them

    Visualizing attention helps us:
    - Verify the model learned the right strategy
    - Debug why certain operations fail
    - Understand what each attention head specializes in
    - Build intuition about transformer internals

Key Functions:
    - extract_attention_weights(): Get attention weights from a forward pass
    - plot_attention_heatmap(): Visualize single attention head
    - plot_all_heads(): Compare all attention heads side by side
    - analyze_attention_patterns(): Automatic pattern detection
"""

import numpy as np


def extract_attention_weights(model, input_tokens):
    """
    Extract attention weights from all layers and heads.

    This requires modifying the model to save attention weights during forward pass.
    For now, we'll return a placeholder structure.

    Args:
        model: Transformer model
        input_tokens: Input sequence (batch_size, seq_len)

    Returns:
        list: Attention weights for each layer
              Format: [layer0_weights, layer1_weights, ...]
              Each layer_weights is shape (num_heads, seq_len, seq_len)

    Note:
        To actually extract attention weights, the MultiHeadAttention class
        needs to save them during forward(). For now, this is a stub.

    TODO: Modify MultiHeadAttention.forward() to save attention weights
    """
    # For now, return a message
    print("⚠ Attention extraction not yet implemented in MultiHeadAttention")
    print("  To enable: modify src/layers/attention.py to save attention weights")

    # Return dummy data for demonstration
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads
    seq_len = input_tokens.shape[1]

    dummy_attention = []
    for layer_idx in range(num_layers):
        # Random attention weights (just for demonstration)
        layer_attention = np.random.rand(num_heads, seq_len, seq_len)
        # Normalize to sum to 1 along last dim (like real attention)
        layer_attention = layer_attention / layer_attention.sum(axis=-1, keepdims=True)
        dummy_attention.append(layer_attention)

    return dummy_attention


def visualize_attention_pattern(attention_weights, input_tokens, token_names=None,
                                head_idx=0, layer_idx=0, title=None):
    """
    Print a text-based visualization of attention weights.

    This is a simple ASCII visualization that works without matplotlib.
    For prettier visualizations, use plot_attention_heatmap() instead.

    Args:
        attention_weights: List of attention weights per layer
        input_tokens: Input sequence
        token_names: Optional list of token names for display
        head_idx: Which attention head to visualize
        layer_idx: Which layer to visualize
        title: Optional title for the visualization

    Example Output:
        Attention Pattern (Layer 0, Head 0)
        ─────────────────────────────────
               Max  (   5   ,   3   ,   9   )
        Max    0.1  0.0 0.4 0.0 0.2 0.0 0.3 0.0
        (      0.1  0.1 0.3 0.0 0.2 0.0 0.2 0.0
        5      0.0  0.0 0.9 0.0 0.1 0.0 0.0 0.0
        ...
    """
    if title is None:
        title = f"Attention Pattern (Layer {layer_idx}, Head {head_idx})"

    print(f"\n{title}")
    print("─" * 60)

    layer_attention = attention_weights[layer_idx]
    head_attention = layer_attention[head_idx]  # Shape: (seq_len, seq_len)

    seq_len = head_attention.shape[0]

    # Create token labels
    if token_names is None:
        labels = [f"T{i}" for i in range(seq_len)]
    else:
        labels = token_names[:seq_len]

    # Print header (columns)
    header = "       " + " ".join(f"{label:4s}" for label in labels)
    print(header)

    # Print each row
    for i, label in enumerate(labels):
        row_str = f"{label:6s} "
        for j in range(seq_len):
            weight = head_attention[i, j]
            # Use intensity to show attention strength
            if weight > 0.5:
                row_str += f"{weight:4.2f} "
            elif weight > 0.2:
                row_str += f"{weight:4.2f} "
            else:
                row_str += f"{weight:4.2f} "
        print(row_str)


def analyze_attention_patterns(attention_weights, input_tokens, operation_name):
    """
    Automatically analyze attention patterns for a given operation.

    This function detects common attention patterns:
    - Positional: Does the model attend to specific positions?
    - Uniform: Does the model attend equally to all positions?
    - Selective: Does the model attend to specific tokens?

    Args:
        attention_weights: Attention weights from extract_attention_weights()
        input_tokens: Input sequence
        operation_name: Name of the operation (e.g., "First", "Max")

    Returns:
        dict: Analysis results with pattern classifications

    Example:
        >>> analysis = analyze_attention_patterns(attn, tokens, "First")
        >>> print(f"Pattern: {analysis['pattern_type']}")
        >>> print(f"Focused on positions: {analysis['focus_positions']}")
    """
    # For "First", we expect strong attention to position of first number
    # For "Max", we expect distributed attention across all numbers

    analysis = {
        'operation': operation_name,
        'pattern_type': 'unknown',
        'focus_positions': [],
        'entropy': 0.0,  # High entropy = uniform attention
    }

    # Get attention from first layer, first head (as example)
    head_attention = attention_weights[0][0]  # Shape: (seq_len, seq_len)

    # For each query position, find where it attends most
    max_attentions = np.argmax(head_attention, axis=1)

    # Compute entropy of attention distribution (how spread out it is)
    eps = 1e-10
    entropy = -np.sum(head_attention * np.log(head_attention + eps), axis=1).mean()
    analysis['entropy'] = float(entropy)

    # High entropy means uniform attention (Max/Min strategy)
    # Low entropy means focused attention (First/Second/Last strategy)
    if entropy > 2.0:
        analysis['pattern_type'] = 'distributed'
    elif entropy < 1.0:
        analysis['pattern_type'] = 'focused'
        # Find which positions are attended to most
        focus_counts = np.bincount(max_attentions, minlength=head_attention.shape[1])
        analysis['focus_positions'] = np.argsort(focus_counts)[::-1][:3].tolist()
    else:
        analysis['pattern_type'] = 'mixed'

    return analysis


def print_attention_summary(model, dataset, num_examples=5):
    """
    Print a summary of attention patterns across multiple examples.

    Args:
        model: Trained model
        dataset: Test dataset
        num_examples: How many examples to analyze
    """
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)

    from vocabluary import VOCAB, detokenize

    for i, (input_indices, answer_indices) in enumerate(dataset[:num_examples]):
        # Prepare input
        max_len = 20
        input_padded = np.array(input_indices + [0] * (max_len - len(input_indices)))
        input_batch = input_padded.reshape(1, -1)

        # Get attention weights
        attention_weights = extract_attention_weights(model, input_batch)

        # Decode tokens for display
        token_names = [detokenize([idx]) for idx in input_indices]

        # Get operation name
        operation_token = input_indices[0]
        operation_name = None
        for name, idx in VOCAB.items():
            if idx == operation_token and name in ['First', 'Second', 'Last', 'Max', 'Min']:
                operation_name = name
                break

        print(f"\nExample {i + 1}: {' '.join(token_names)}")
        print(f"Operation: {operation_name}")

        # Analyze patterns
        if attention_weights:
            analysis = analyze_attention_patterns(attention_weights, input_batch[0], operation_name)
            print(f"  Pattern Type: {analysis['pattern_type']}")
            print(f"  Attention Entropy: {analysis['entropy']:.2f}")
            if analysis['focus_positions']:
                print(f"  Focused on positions: {analysis['focus_positions']}")


# ============================================================================
# Matplotlib-based visualizations (optional, requires matplotlib)
# ============================================================================

def plot_attention_heatmap(attention_weights, input_tokens, token_names=None,
                          layer_idx=0, head_idx=0, save_path=None):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights from extract_attention_weights()
        input_tokens: Input sequence
        token_names: Optional token names for axes labels
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize
        save_path: Where to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        print("  Install with: pip install matplotlib")
        return

    head_attention = attention_weights[layer_idx][head_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(head_attention, cmap='viridis', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    seq_len = head_attention.shape[0]
    if token_names is None:
        token_names = [f"T{i}" for i in range(seq_len)]

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(token_names, rotation=45, ha='right')
    ax.set_yticklabels(token_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    # Labels and title
    ax.set_xlabel('Key Position (attending to)', fontsize=12)
    ax.set_ylabel('Query Position (attending from)', fontsize=12)
    ax.set_title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved attention heatmap: {save_path}")

    plt.show()


def plot_all_heads(attention_weights, input_tokens, token_names=None,
                  layer_idx=0, save_path=None):
    """
    Plot all attention heads for a single layer side by side.

    This lets us compare what each head is learning.

    Args:
        attention_weights: Attention weights from extract_attention_weights()
        input_tokens: Input sequence
        token_names: Optional token names
        layer_idx: Which layer to visualize
        save_path: Where to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        return

    layer_attention = attention_weights[layer_idx]
    num_heads = layer_attention.shape[0]

    # Create subplots grid
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if num_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    seq_len = layer_attention.shape[1]
    if token_names is None:
        token_names = [f"T{i}" for i in range(seq_len)]

    for head_idx in range(num_heads):
        row = head_idx // cols
        col = head_idx % cols
        ax = axes[row, col]

        head_attention = layer_attention[head_idx]

        # Plot heatmap
        im = ax.imshow(head_attention, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Key', fontsize=9)
        ax.set_ylabel('Query', fontsize=9)

        # Simplified ticks
        ax.set_xticks(range(0, seq_len, max(1, seq_len // 5)))
        ax.set_yticks(range(0, seq_len, max(1, seq_len // 5)))

    # Hide extra subplots
    for idx in range(num_heads, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle(f'All Attention Heads - Layer {layer_idx}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved all heads visualization: {save_path}")

    plt.show()


def plot_attention_evolution(model, example, num_layers=None):
    """
    Show how attention patterns evolve across layers.

    Args:
        model: Trained model
        example: Single example (input_indices, answer_indices)
        num_layers: Number of layers to show (default: all)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        return

    from vocabluary import detokenize

    input_indices, _ = example

    # Prepare input
    max_len = 20
    input_padded = np.array(input_indices + [0] * (max_len - len(input_indices)))
    input_batch = input_padded.reshape(1, -1)

    # Extract attention
    attention_weights = extract_attention_weights(model, input_batch)

    if num_layers is None:
        num_layers = len(attention_weights)

    # Get token names
    token_names = [detokenize([idx]) for idx in input_indices[:8]]  # Show first 8 tokens

    # Plot first head of each layer
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
    if num_layers == 1:
        axes = [axes]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        # Get first head attention for this layer
        head_attention = attention_weights[layer_idx][0][:8, :8]  # Show 8x8 region

        im = ax.imshow(head_attention, cmap='viridis', aspect='auto', vmin=0, vmax=1)

        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key', fontsize=10)
        if layer_idx == 0:
            ax.set_ylabel('Query', fontsize=10)

        ax.set_xticks(range(len(token_names)))
        ax.set_yticks(range(len(token_names)))
        ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(token_names, fontsize=8)

    plt.suptitle('Attention Evolution Across Layers (Head 0)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION MODULE TEST")
    print("=" * 60)

    # Test 1: Dummy attention visualization
    print("\n1. Testing attention extraction (with dummy data):")
    print("-" * 60)

    # Create dummy model-like structure
    class DummyBlock:
        def __init__(self):
            class DummyAttention:
                num_heads = 4
            self.attention = DummyAttention()

    class DummyModel:
        def __init__(self):
            self.blocks = [DummyBlock(), DummyBlock()]

    dummy_model = DummyModel()
    dummy_input = np.array([[15, 17, 7, 19, 5, 18, 0, 0]])

    attention_weights = extract_attention_weights(dummy_model, dummy_input)
    print(f"  ✓ Extracted attention for {len(attention_weights)} layers")
    print(f"  ✓ Layer 0 shape: {attention_weights[0].shape}")  # (num_heads, seq_len, seq_len)

    # Test 2: Text-based visualization
    print("\n2. Text-based Attention Visualization:")
    print("-" * 60)

    token_names = ['Max', '(', '5', ',', '3', ',', '9', ')']
    visualize_attention_pattern(attention_weights, dummy_input, token_names,
                               head_idx=0, layer_idx=0)

    # Test 3: Pattern analysis
    print("\n3. Attention Pattern Analysis:")
    print("-" * 60)

    analysis = analyze_attention_patterns(attention_weights, dummy_input[0], "Max")
    print(f"  Operation: {analysis['operation']}")
    print(f"  Pattern Type: {analysis['pattern_type']}")
    print(f"  Entropy: {analysis['entropy']:.2f}")
    print(f"  (High entropy = distributed attention, Low = focused)")

    print("\n" + "=" * 60)
    print("VISUALIZATION MODULE READY")
    print("=" * 60)
    print("\nNote: To enable real attention extraction:")
    print("  1. Modify MultiHeadAttention.forward() to save attention weights")
    print("  2. Add: self.last_attention_weights = attention_weights")
    print("  3. Then extract via: model.blocks[i].attention.last_attention_weights")
    print("\nFor prettier plots, install matplotlib:")
    print("  pip install matplotlib")
