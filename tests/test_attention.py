"""
Comprehensive test suite for attention mechanisms.

Tests cover:
1. Scaled dot-product attention
   - Forward pass shape verification
   - Attention weights sum to 1
   - Output is weighted combination of values
   - Masking works correctly

2. Multi-head attention
   - Forward pass shape verification
   - Multiple heads learn independently
   - Dimension splitting and combining
   - Gradient flow (simplified check)

These tests ensure the attention mechanism is correctly implemented
and ready to be integrated into the transformer architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from layers import MultiHeadAttention, scaled_dot_product_attention


def test_scaled_dot_product_attention_shapes():
    """Test that scaled dot-product attention produces correct output shapes."""
    print("Testing scaled_dot_product_attention shapes...")

    batch_size = 2
    seq_len = 10
    d_k = 64
    d_v = 64

    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_v), \
        f"Expected output shape {(batch_size, seq_len, d_v)}, got {output.shape}"

    # Check attention weights shape
    assert attention_weights.shape == (batch_size, seq_len, seq_len), \
        f"Expected attention weights shape {(batch_size, seq_len, seq_len)}, got {attention_weights.shape}"

    print("✓ Scaled dot-product attention shapes are correct")


def test_attention_weights_sum_to_one():
    """Test that attention weights form a valid probability distribution."""
    print("Testing attention weights sum to 1...")

    batch_size = 3
    seq_len = 5
    d_k = 32

    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    _, attention_weights = scaled_dot_product_attention(Q, K, V)

    # Each row of attention weights should sum to 1 (valid probability distribution)
    row_sums = np.sum(attention_weights, axis=-1)
    assert np.allclose(row_sums, 1.0, rtol=1e-5), \
        f"Attention weights should sum to 1, got {row_sums}"

    # All weights should be non-negative
    assert np.all(attention_weights >= 0), \
        "Attention weights should be non-negative"

    # All weights should be at most 1
    assert np.all(attention_weights <= 1), \
        "Attention weights should be at most 1"

    print("✓ Attention weights form valid probability distributions")


def test_attention_with_masking():
    """Test that masking correctly prevents attention to certain positions."""
    print("Testing attention masking...")

    batch_size = 1
    seq_len = 4
    d_k = 8

    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # Create mask: don't attend to last two positions
    mask = np.ones((batch_size, seq_len, seq_len))
    mask[:, :, -2:] = 0  # Mask out last two positions

    _, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    # Attention to masked positions should be near zero
    masked_attention = attention_weights[:, :, -2:]
    assert np.allclose(masked_attention, 0.0, atol=1e-6), \
        f"Masked positions should have near-zero attention, got max {np.max(masked_attention)}"

    # Attention to non-masked positions should still sum to 1
    non_masked_attention = attention_weights[:, :, :-2]
    row_sums = np.sum(non_masked_attention, axis=-1)
    assert np.allclose(row_sums, 1.0, rtol=1e-5), \
        "Non-masked attention weights should still sum to 1"

    print("✓ Attention masking works correctly")


def test_attention_output_is_weighted_values():
    """Test that attention output is correctly computed as weighted sum of values."""
    print("Testing attention output computation...")

    batch_size = 1
    seq_len = 3
    d_k = 4
    d_v = 4

    # Use simple values for manual verification
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.array([[[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]]])  # Identity-like values

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    # Manually compute expected output
    expected_output = np.matmul(attention_weights, V)

    assert np.allclose(output, expected_output, rtol=1e-5), \
        "Output should be weighted sum of values using attention weights"

    print("✓ Attention output is correctly computed")


def test_multihead_attention_shapes():
    """Test that multi-head attention produces correct output shapes."""
    print("Testing MultiHeadAttention shapes...")

    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    output, attention_weights = mha.forward(x)

    # Output should have same shape as input
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"Expected output shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    # Attention weights should have shape (batch, num_heads, seq_len, seq_len)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Expected attention shape {(batch_size, num_heads, seq_len, seq_len)}, got {attention_weights.shape}"

    print("✓ MultiHeadAttention shapes are correct")


def test_multihead_attention_embed_dim_divisibility():
    """Test that MultiHeadAttention requires embed_dim divisible by num_heads."""
    print("Testing embed_dim divisibility requirement...")

    # This should work
    try:
        mha = MultiHeadAttention(embed_dim=64, num_heads=4)
        print("✓ Accepts embed_dim=64, num_heads=4 (divisible)")
    except AssertionError:
        print("✗ Failed to create valid MultiHeadAttention")
        raise

    # This should fail
    try:
        mha = MultiHeadAttention(embed_dim=64, num_heads=5)
        print("✗ Should have rejected embed_dim=64, num_heads=5 (not divisible)")
        raise AssertionError("Should have failed with non-divisible dimensions")
    except AssertionError as e:
        if "must be divisible" in str(e):
            print("✓ Correctly rejects non-divisible dimensions")
        else:
            raise


def test_multihead_attention_head_independence():
    """Test that different heads can learn different patterns."""
    print("Testing multi-head independence...")

    embed_dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 8

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    _, attention_weights = mha.forward(x)

    # Different heads should produce different attention patterns
    head_0 = attention_weights[0, 0, :, :]  # First head
    head_1 = attention_weights[0, 1, :, :]  # Second head

    # Heads should not be identical (very unlikely with random initialization)
    assert not np.allclose(head_0, head_1, rtol=1e-3), \
        "Different heads should produce different attention patterns"

    # Each head's attention weights should still sum to 1
    for h in range(num_heads):
        head_weights = attention_weights[0, h, :, :]
        row_sums = np.sum(head_weights, axis=-1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5), \
            f"Head {h} attention weights should sum to 1"

    print("✓ Multi-head attention heads operate independently")


def test_multihead_attention_forward_backward():
    """Test that gradients flow through multi-head attention."""
    print("Testing MultiHeadAttention forward and backward pass...")

    embed_dim = 32
    num_heads = 4
    batch_size = 2
    seq_len = 5

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output, _ = mha.forward(x)

    # Create dummy gradient
    grad_output = np.random.randn(*output.shape)

    # Backward pass
    grad_input = mha.backward(grad_output)

    # Check gradient shape
    assert grad_input.shape == x.shape, \
        f"Expected grad_input shape {x.shape}, got {grad_input.shape}"

    # Check that parameter gradients are computed
    assert mha.grad_W_q is not None, "grad_W_q should be computed"
    assert mha.grad_W_k is not None, "grad_W_k should be computed"
    assert mha.grad_W_v is not None, "grad_W_v should be computed"
    assert mha.grad_W_o is not None, "grad_W_o should be computed"

    # Check gradient shapes
    assert mha.grad_W_q.shape == mha.W_q.shape, "grad_W_q shape mismatch"
    assert mha.grad_W_k.shape == mha.W_k.shape, "grad_W_k shape mismatch"
    assert mha.grad_W_v.shape == mha.W_v.shape, "grad_W_v shape mismatch"
    assert mha.grad_W_o.shape == mha.W_o.shape, "grad_W_o shape mismatch"

    # Check that gradients are not all zeros (information flows)
    assert not np.allclose(grad_input, 0), "grad_input should not be all zeros"
    assert not np.allclose(mha.grad_W_q, 0), "grad_W_q should not be all zeros"

    print("✓ MultiHeadAttention backward pass computes gradients correctly")


def test_multihead_attention_parameters():
    """Test that MultiHeadAttention correctly reports its parameters."""
    print("Testing MultiHeadAttention parameter management...")

    embed_dim = 64
    num_heads = 4

    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Get parameters
    params = mha.get_parameters()

    # Should have 4 parameter groups (Q, K, V, O)
    assert len(params) == 4, f"Expected 4 parameter groups, got {len(params)}"

    # Before backward pass, gradients should be None
    for param, grad in params:
        assert param is not None, "Parameter should not be None"
        assert grad is None, "Gradient should be None before backward pass"

    # Test zero_grad
    mha.grad_W_q = np.ones_like(mha.W_q)  # Set dummy gradient
    mha.zero_grad()
    assert mha.grad_W_q is None, "zero_grad should reset gradients to None"

    print("✓ MultiHeadAttention parameter management works correctly")


def test_attention_no_nans():
    """Test that attention produces no NaN or Inf values."""
    print("Testing attention numerical stability...")

    # Test with various edge cases
    test_cases = [
        # Normal case
        (np.random.randn(2, 5, 32), np.random.randn(2, 5, 32), np.random.randn(2, 5, 32)),
        # Large values
        (np.random.randn(2, 5, 32) * 10, np.random.randn(2, 5, 32) * 10, np.random.randn(2, 5, 32)),
        # Small values
        (np.random.randn(2, 5, 32) * 0.01, np.random.randn(2, 5, 32) * 0.01, np.random.randn(2, 5, 32)),
    ]

    for i, (Q, K, V) in enumerate(test_cases):
        output, attention_weights = scaled_dot_product_attention(Q, K, V)

        assert not np.any(np.isnan(output)), f"Test case {i}: output contains NaN"
        assert not np.any(np.isinf(output)), f"Test case {i}: output contains Inf"
        assert not np.any(np.isnan(attention_weights)), f"Test case {i}: attention weights contain NaN"
        assert not np.any(np.isinf(attention_weights)), f"Test case {i}: attention weights contain Inf"

    print("✓ Attention is numerically stable (no NaN or Inf)")


def test_multihead_attention_with_different_configs():
    """Test MultiHeadAttention with various configuration combinations."""
    print("Testing MultiHeadAttention with different configurations...")

    configs = [
        (64, 1),   # Single head
        (64, 4),   # Standard config
        (64, 8),   # Many heads
        (128, 8),  # Larger embedding
        (32, 2),   # Smaller embedding
    ]

    for embed_dim, num_heads in configs:
        mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = np.random.randn(2, 10, embed_dim)

        output, attention_weights = mha.forward(x)

        # Verify shapes
        assert output.shape == (2, 10, embed_dim), \
            f"Config ({embed_dim}, {num_heads}): wrong output shape"
        assert attention_weights.shape == (2, num_heads, 10, 10), \
            f"Config ({embed_dim}, {num_heads}): wrong attention shape"

    print("✓ MultiHeadAttention works with various configurations")


# Run all tests
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Attention Mechanism Tests")
    print("="*60 + "\n")

    try:
        # Scaled dot-product attention tests
        test_scaled_dot_product_attention_shapes()
        test_attention_weights_sum_to_one()
        test_attention_with_masking()
        test_attention_output_is_weighted_values()
        test_attention_no_nans()

        print()

        # Multi-head attention tests
        test_multihead_attention_shapes()
        test_multihead_attention_embed_dim_divisibility()
        test_multihead_attention_head_independence()
        test_multihead_attention_forward_backward()
        test_multihead_attention_parameters()
        test_multihead_attention_with_different_configs()

        print("\n" + "="*60)
        print("All attention tests passed! ✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
