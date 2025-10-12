"""
Positional Encoding Module

This module implements sinusoidal positional encoding for transformers.

The Problem:
    Transformers process all tokens in parallel (unlike RNNs which process sequentially).
    This efficiency comes at a cost: the model has no inherent notion of token order.
    Without positional information, "the cat sat" and "sat cat the" would look identical!

The Solution - Positional Encoding:
    Add position-dependent patterns to embeddings so the model can distinguish positions.

Why Sinusoidal Encoding?
    The original Transformer paper (Vaswani et al., 2017) used sine and cosine functions
    because they have several desirable properties:
    1. Bounded values [-1, 1] - prevents overwhelming the embeddings
    2. Unique encoding for each position - no two positions have the same pattern
    3. Relative position information - sin(a+b) can be expressed using sin(a) and sin(b)
    4. Generalizes to longer sequences - works for any sequence length
    5. No learned parameters - one less thing to train

Mathematical Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos is the position in the sequence (0, 1, 2, ...)
    - i is the dimension index (0, 1, 2, ..., d_model/2)
    - Even dimensions use sine, odd dimensions use cosine

Intuition:
    Think of it like a unique fingerprint for each position:
    - Low-frequency dimensions change slowly (capture long-range patterns)
    - High-frequency dimensions change quickly (capture local patterns)
    - Together they create a unique encoding for each position

Alternative Approaches:
    - Learned positional embeddings: Train position vectors like token embeddings
      Pros: Can adapt to data, Cons: Doesn't generalize beyond training length
    - Relative positional encoding: Encode distances between positions
      Pros: Better for some tasks, Cons: More complex
    - RoPE (Rotary Position Embedding): Modern alternative used in LLaMA, GPT-NeoX
      Pros: Better long-range performance, Cons: More complex implementation
"""

import numpy as np


class PositionalEncoding:
    """
    Positional Encoding using sinusoidal functions.

    Adds position information to embeddings so the model knows token order.
    Uses fixed (not learned) sinusoidal patterns that generalize to any sequence length.

    The encoding is added to the embedding vectors, enriching them with positional context.
    After adding positional encoding, the model can distinguish between identical tokens
    at different positions.

    Args:
        max_seq_len (int): Maximum sequence length to precompute encodings for
        embed_dim (int): Dimension of embeddings (must match embedding layer)

    Attributes:
        encodings (np.ndarray): Precomputed positional encodings,
                               shape (max_seq_len, embed_dim)

    Example:
        >>> pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
        >>> embedded = np.random.randn(2, 10, 64)  # (batch, seq_len, embed_dim)
        >>> with_position = pos_enc.forward(embedded)  # adds positional information
    """

    def __init__(self, max_seq_len, embed_dim):
        """
        Initialize positional encoding.

        We precompute all positional encodings up to max_seq_len for efficiency.
        This is done once during initialization rather than recomputing every forward pass.

        Why Precompute?
            Positional encodings are deterministic (not learned), so we can calculate
            them once and reuse them. This saves computation during training.

        Args:
            max_seq_len (int): Maximum sequence length to support
            embed_dim (int): Embedding dimension (must be even for sin/cos pairing)
        """
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Precompute all positional encodings
        # Shape: (max_seq_len, embed_dim)
        self.encodings = self._create_positional_encodings()

    def _create_positional_encodings(self):
        """
        Create sinusoidal positional encodings.

        Implementation follows the formula from "Attention Is All You Need":
            PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Why This Formula?
            The 10000^(2i/d_model) term creates different frequencies for different dimensions:
            - Lower dimensions (i=0) have high frequency (change quickly with position)
            - Higher dimensions (i=d/2) have low frequency (change slowly with position)

            This multi-scale encoding allows the model to attend to both:
            - Local patterns (high-frequency dimensions)
            - Long-range dependencies (low-frequency dimensions)

        Why Sin and Cos?
            Using both sine and cosine allows the model to learn relative positions:
            - sin(pos + k) can be expressed as a linear function of sin(pos) and cos(pos)
            - This helps the model understand that position 5 is 3 steps from position 2

        Implementation Details:
            We compute div_term = 1 / (10000^(2i/d_model)) using exp and log for
            numerical stability: exp(log(10000) * (-2i/d_model))

        Returns:
            np.ndarray: Positional encoding matrix of shape (max_seq_len, embed_dim)
        """
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape: (max_seq_len, 1) - one column for broadcasting
        position = np.arange(self.max_seq_len)[:, np.newaxis]

        # Create dimension indices for even positions: [0, 2, 4, ..., embed_dim-2]
        # We only need half the dimensions because sin and cos will fill both even and odd
        # Compute the divisor: 10000^(2i/d_model)
        # Using exp/log for numerical stability instead of direct power
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))

        # Initialize encoding matrix
        encodings = np.zeros((self.max_seq_len, self.embed_dim))

        # Apply sin to even indices (0, 2, 4, ...)
        # Broadcasting: (max_seq_len, 1) * (embed_dim/2,) -> (max_seq_len, embed_dim/2)
        encodings[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd indices (1, 3, 5, ...)
        encodings[:, 1::2] = np.cos(position * div_term)

        return encodings

    def forward(self, embedded_tokens):
        """
        Add positional encodings to embedded tokens.

        The positional encoding is added (not concatenated) to preserve the embedding
        dimension. This addition allows the model to learn how to use both content
        (from embeddings) and position (from encoding) information.

        Why Addition Instead of Concatenation?
            - Addition: Preserves dimension, mixes position with content
            - Concatenation: Doubles dimension, keeps them separate
            Addition is preferred because:
            1. Keeps model size smaller (no dimension increase)
            2. Forces tight integration of position and content
            3. Works well in practice (standard in transformers)

        Args:
            embedded_tokens (np.ndarray): Embedded tokens of shape
                                         (batch_size, seq_len, embed_dim)

        Returns:
            np.ndarray: Tokens with positional encoding added, same shape as input

        Raises:
            ValueError: If seq_len > max_seq_len or embed_dim doesn't match
        """
        batch_size, seq_len, embed_dim = embedded_tokens.shape

        # Validation checks
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        if embed_dim != self.embed_dim:
            raise ValueError(f"Embedding dimension {embed_dim} does not match {self.embed_dim}")

        # Get encodings for this sequence length
        # Shape: (seq_len, embed_dim)
        pos_encodings = self.encodings[:seq_len, :]

        # Add to embedded tokens
        # Broadcasting automatically handles the batch dimension:
        # (batch_size, seq_len, embed_dim) + (seq_len, embed_dim)
        # -> (batch_size, seq_len, embed_dim)
        output = embedded_tokens + pos_encodings

        return output

    def backward(self, grad_output):
        """
        Backward pass for positional encoding.

        Since positional encodings are fixed (not learned), they have no parameters
        to update. The gradient simply flows through unchanged.

        Why No Gradient Computation?
            The positional encoding is not learned (no parameters), so there's nothing
            to update. We just pass the gradient through to the embedding layer.

        This is similar to how ReLU passes gradient through for positive values -
        the operation affects forward pass but doesn't require its own parameter updates.

        Args:
            grad_output (np.ndarray): Gradient from next layer

        Returns:
            np.ndarray: Same gradient (unchanged) to pass to previous layer
        """
        # Gradient flows through unchanged
        # Addition operation has gradient of 1, so: dL/d(x+p) = dL/dx + dL/dp
        # Since p is not learnable, we only care about dL/dx, which equals grad_output
        return grad_output

    def get_parameters(self):
        """
        Get trainable parameters.

        Returns:
            list: Empty list (positional encodings are fixed, not learned)

        Note:
            Alternative implementations might use learned positional embeddings,
            which would be returned here. We use fixed sinusoidal encodings.
        """
        return []

    def zero_grad(self):
        """
        Reset gradients.

        Does nothing since there are no parameters and therefore no gradients.
        Included for consistency with other layer interfaces.
        """
        pass

    def __repr__(self):
        """String representation for debugging."""
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"
