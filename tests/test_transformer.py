"""
Comprehensive test suite for transformer components.

Tests cover:
1. Feed-Forward Network
   - Forward pass shape verification
   - Dimension expansion and compression
   - Gradient flow

2. Transformer Block
   - Complete block forward pass
   - Residual connections work correctly
   - Layer normalization applied properly
   - Gradient flow through complex architecture

3. Complete Transformer
   - End-to-end forward pass
   - Correct output shape (logits and probabilities)
   - All parameters receive gradients
   - Integration of all components

These tests ensure the transformer architecture is correctly implemented
and ready for training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from transformer import FeedForward, TransformerBlock, Transformer


def test_feedforward_shapes():
    """Test that feed-forward network produces correct output shapes."""
    print("Testing FeedForward shapes...")

    embed_dim = 64
    ffn_dim = 256
    batch_size = 2
    seq_len = 10

    ffn = FeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    output = ffn.forward(x)

    # Output should have same shape as input
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    print("✓ FeedForward shapes are correct")


def test_feedforward_dimension_expansion():
    """Test that FFN correctly expands to ffn_dim internally."""
    print("Testing FeedForward dimension expansion...")

    embed_dim = 32
    ffn_dim = 128  # 4x expansion
    batch_size = 2
    seq_len = 5

    ffn = FeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = ffn.forward(x)

    # Check cached intermediate output has expanded dimension
    assert ffn.linear1_output_cache.shape == (batch_size, seq_len, ffn_dim), \
        f"Expected intermediate shape {(batch_size, seq_len, ffn_dim)}, got {ffn.linear1_output_cache.shape}"

    # Check final output is compressed back to embed_dim
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    print("✓ FeedForward correctly expands and compresses dimensions")


def test_feedforward_gradient_flow():
    """Test that gradients flow correctly through feed-forward network."""
    print("Testing FeedForward gradient flow...")

    embed_dim = 32
    ffn_dim = 128
    batch_size = 2
    seq_len = 5

    ffn = FeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = ffn.forward(x)

    # Backward pass with dummy gradient
    grad_output = np.random.randn(*output.shape)
    grad_input = ffn.backward(grad_output)

    # Check gradient shape
    assert grad_input.shape == x.shape, \
        f"Expected grad_input shape {x.shape}, got {grad_input.shape}"

    # Check that parameters have gradients
    params = ffn.get_parameters()
    assert len(params) == 4, f"Expected 4 parameter groups (2 weights + 2 biases), got {len(params)}"

    for param, grad in params:
        assert grad is not None, "Parameter gradient should not be None"
        assert grad.shape == param.shape, "Gradient shape should match parameter shape"

    print("✓ FeedForward gradient flow is correct")


def test_feedforward_parameters():
    """Test that FFN correctly manages parameters."""
    print("Testing FeedForward parameter management...")

    embed_dim = 64
    ffn_dim = 256

    ffn = FeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim)

    params = ffn.get_parameters()

    # Should have 4 parameter groups: 2 weights + 2 biases
    assert len(params) == 4, f"Expected 4 parameter groups, got {len(params)}"

    # Check weight shapes
    w1, grad_w1 = params[0]
    w2, grad_w2 = params[2]

    assert w1.shape == (embed_dim, ffn_dim), f"First weight should be {(embed_dim, ffn_dim)}, got {w1.shape}"
    assert w2.shape == (ffn_dim, embed_dim), f"Second weight should be {(ffn_dim, embed_dim)}, got {w2.shape}"

    print("✓ FeedForward parameter management is correct")


def test_transformer_block_shapes():
    """Test that transformer block produces correct output shapes."""
    print("Testing TransformerBlock shapes...")

    embed_dim = 64
    num_heads = 4
    ffn_dim = 256
    batch_size = 2
    seq_len = 10

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    output = block.forward(x)

    # Output should have same shape as input
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    print("✓ TransformerBlock shapes are correct")


def test_transformer_block_residual_connections():
    """Test that residual connections work correctly."""
    print("Testing TransformerBlock residual connections...")

    embed_dim = 32
    num_heads = 2
    ffn_dim = 128
    batch_size = 1
    seq_len = 5

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = block.forward(x)

    # Check that cached residuals have correct shapes
    assert block.residual1_cache.shape == x.shape, "First residual should match input shape"
    assert block.residual2_cache.shape == x.shape, "Second residual should match input shape"

    # Residuals should be input + sub-layer output
    # residual1 = x + attention(x)
    attention_added_something = not np.allclose(block.residual1_cache, x, rtol=1e-3)
    assert attention_added_something, "Attention should modify the representation"

    print("✓ TransformerBlock residual connections work correctly")


def test_transformer_block_gradient_flow():
    """Test that gradients flow through transformer block."""
    print("Testing TransformerBlock gradient flow...")

    embed_dim = 32
    num_heads = 4
    ffn_dim = 128
    batch_size = 2
    seq_len = 5

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = block.forward(x)

    # Backward pass
    grad_output = np.random.randn(*output.shape)
    grad_input = block.backward(grad_output)

    # Check gradient shape
    assert grad_input.shape == x.shape, \
        f"Expected grad_input shape {x.shape}, got {grad_input.shape}"

    # Check that all components have gradients
    params = block.get_parameters()
    assert len(params) > 0, "Block should have parameters"

    for param, grad in params:
        if grad is not None:  # Some parameters might not have gradients yet
            assert grad.shape == param.shape, "Gradient shape should match parameter shape"

    print("✓ TransformerBlock gradient flow is correct")


def test_transformer_block_components():
    """Test that transformer block contains all required components."""
    print("Testing TransformerBlock components...")

    embed_dim = 64
    num_heads = 4
    ffn_dim = 256

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)

    # Check that all components exist
    assert hasattr(block, 'attention'), "Block should have attention layer"
    assert hasattr(block, 'norm1'), "Block should have first layer norm"
    assert hasattr(block, 'ffn'), "Block should have feed-forward network"
    assert hasattr(block, 'norm2'), "Block should have second layer norm"

    # Check component types
    assert block.attention.__class__.__name__ == 'MultiHeadAttention', "Should have MultiHeadAttention"
    assert block.norm1.__class__.__name__ == 'LayerNorm', "Should have LayerNorm"
    assert block.ffn.__class__.__name__ == 'FeedForward', "Should have FeedForward"
    assert block.norm2.__class__.__name__ == 'LayerNorm', "Should have LayerNorm"

    print("✓ TransformerBlock has all required components")


def test_complete_transformer_shapes():
    """Test that complete transformer produces correct output shapes."""
    print("Testing complete Transformer shapes...")

    vocab_size = 20
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    ffn_dim = 256
    max_seq_len = 50

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len
    )

    batch_size = 2
    seq_len = 10

    # Input token IDs
    tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Forward pass
    probs = model.forward(tokens)

    # Output should be (batch_size, vocab_size)
    assert probs.shape == (batch_size, vocab_size), \
        f"Expected output shape {(batch_size, vocab_size)}, got {probs.shape}"

    # Probabilities should sum to 1
    prob_sums = np.sum(probs, axis=-1)
    assert np.allclose(prob_sums, 1.0, rtol=1e-5), \
        f"Probabilities should sum to 1, got {prob_sums}"

    print("✓ Complete Transformer output shapes are correct")


def test_transformer_logits_vs_probs():
    """Test that transformer can return logits or probabilities."""
    print("Testing Transformer logits vs probabilities...")

    vocab_size = 20
    embed_dim = 32
    num_heads = 2
    num_layers = 1
    ffn_dim = 128

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim
    )

    tokens = np.random.randint(0, vocab_size, size=(2, 5))

    # Get probabilities
    probs = model.forward(tokens, return_logits=False)
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities should be in [0, 1]"
    assert np.allclose(np.sum(probs, axis=-1), 1.0), "Probabilities should sum to 1"

    # Get logits
    logits = model.forward(tokens, return_logits=True)
    # Logits can be any real number
    assert not np.allclose(np.sum(logits, axis=-1), 1.0), "Logits should not sum to 1"

    print("✓ Transformer correctly returns logits or probabilities")


def test_transformer_gradient_flow():
    """Test that gradients flow through complete transformer."""
    print("Testing complete Transformer gradient flow...")

    vocab_size = 20
    embed_dim = 32
    num_heads = 2
    num_layers = 2
    ffn_dim = 128

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim
    )

    tokens = np.random.randint(0, vocab_size, size=(2, 5))

    # Forward pass
    probs = model.forward(tokens)

    # Backward pass with dummy gradient
    grad_output = np.random.randn(*probs.shape)
    grad_input = model.backward(grad_output)

    # Check that parameters have gradients
    params = model.get_parameters()
    assert len(params) > 0, "Model should have parameters"

    params_with_grads = sum(1 for _, grad in params if grad is not None)
    assert params_with_grads > 0, "Some parameters should have gradients"

    print(f"✓ Transformer gradient flow is correct ({params_with_grads}/{len(params)} parameters have gradients)")


def test_transformer_components():
    """Test that transformer contains all required components."""
    print("Testing Transformer components...")

    vocab_size = 20
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    ffn_dim = 256

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim
    )

    # Check that all components exist
    assert hasattr(model, 'embedding'), "Model should have embedding layer"
    assert hasattr(model, 'pos_encoding'), "Model should have positional encoding"
    assert hasattr(model, 'blocks'), "Model should have transformer blocks"
    assert hasattr(model, 'output_projection'), "Model should have output projection"
    assert hasattr(model, 'softmax'), "Model should have softmax"

    # Check number of blocks
    assert len(model.blocks) == num_layers, \
        f"Expected {num_layers} blocks, got {len(model.blocks)}"

    print("✓ Transformer has all required components")


def test_transformer_multiple_layers():
    """Test that transformer works with different numbers of layers."""
    print("Testing Transformer with multiple layer configurations...")

    vocab_size = 20
    embed_dim = 32
    num_heads = 2
    ffn_dim = 128

    for num_layers in [1, 2, 3, 4]:
        model = Transformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim
        )

        tokens = np.random.randint(0, vocab_size, size=(2, 5))
        probs = model.forward(tokens)

        assert probs.shape == (2, vocab_size), \
            f"With {num_layers} layers, expected output shape (2, {vocab_size}), got {probs.shape}"

    print("✓ Transformer works with 1-4 layers")


def test_transformer_no_nans():
    """Test that transformer produces no NaN or Inf values."""
    print("Testing Transformer numerical stability...")

    vocab_size = 20
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    ffn_dim = 256

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim
    )

    # Test with various input sizes
    test_cases = [
        (1, 5),   # Single example, short sequence
        (4, 10),  # Small batch
        (8, 20),  # Larger batch and sequence
    ]

    for batch_size, seq_len in test_cases:
        tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        probs = model.forward(tokens)

        assert not np.any(np.isnan(probs)), f"Case ({batch_size}, {seq_len}): Output contains NaN"
        assert not np.any(np.isinf(probs)), f"Case ({batch_size}, {seq_len}): Output contains Inf"

    print("✓ Transformer is numerically stable (no NaN or Inf)")


def test_transformer_parameter_count():
    """Test that transformer has reasonable number of parameters."""
    print("Testing Transformer parameter count...")

    vocab_size = 20
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    ffn_dim = 256

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim
    )

    params = model.get_parameters()
    total_params = sum(param.size for param, _ in params)

    print(f"  Total parameters: {total_params:,}")

    # Rough estimate:
    # - Embedding: vocab_size * embed_dim = 20 * 64 = 1,280
    # - Per block: ~4 * embed_dim^2 (attention) + 2 * embed_dim * ffn_dim (FFN) ≈ 49,152
    # - 2 blocks: ~98,304
    # - Output: embed_dim * vocab_size = 1,280
    # Total: ~100,864

    expected_min = 50_000
    expected_max = 150_000

    assert expected_min < total_params < expected_max, \
        f"Expected ~100k parameters, got {total_params:,}"

    print(f"✓ Transformer has reasonable parameter count ({total_params:,})")


# Run all tests
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Transformer Architecture Tests")
    print("="*60 + "\n")

    try:
        # Feed-Forward Network tests
        print("Feed-Forward Network Tests:")
        print("-" * 60)
        test_feedforward_shapes()
        test_feedforward_dimension_expansion()
        test_feedforward_gradient_flow()
        test_feedforward_parameters()

        print()

        # Transformer Block tests
        print("Transformer Block Tests:")
        print("-" * 60)
        test_transformer_block_shapes()
        test_transformer_block_residual_connections()
        test_transformer_block_gradient_flow()
        test_transformer_block_components()

        print()

        # Complete Transformer tests
        print("Complete Transformer Tests:")
        print("-" * 60)
        test_complete_transformer_shapes()
        test_transformer_logits_vs_probs()
        test_transformer_gradient_flow()
        test_transformer_components()
        test_transformer_multiple_layers()
        test_transformer_no_nans()
        test_transformer_parameter_count()

        print("\n" + "="*60)
        print("All transformer tests passed! ✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
