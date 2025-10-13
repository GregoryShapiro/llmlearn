"""
Embedding Layer Module

This module implements token embedding from scratch using NumPy.

The embedding layer is a fundamental component in neural language models that converts
discrete token IDs (integers) into continuous vector representations (embeddings).

Key Concepts:
    - Each token in the vocabulary gets its own learnable vector
    - The embedding matrix has shape (vocab_size, embed_dim)
    - Looking up an embedding is just indexing into this matrix
    - During training, embeddings are learned to capture semantic meaning

Why Embeddings?
    Neural networks can't directly process discrete symbols like words or tokens.
    Embeddings provide a way to represent tokens as dense vectors in a continuous space,
    where semantically similar tokens can be positioned close to each other.

Implementation Details:
    - Small random initialization (0.01 * randn) to break symmetry
    - Gradient accumulation is needed because the same token can appear multiple times
    - No activation function (linear transformation)
"""

import numpy as np
from typing import List, Tuple, Optional


class Embedding:
    """
    Embedding layer that converts token indices to dense vectors.

    This is essentially a lookup table where each token ID maps to a learned vector.
    For example, if token 15 represents 'Max', it might map to [0.2, -0.5, 0.8, ..., 0.1].

    The embedding layer is typically the first layer in a language model, converting
    discrete token IDs into continuous representations that the network can process.

    Mathematical Formulation:
        Given token index i, return embedding_matrix[i]
        This is equivalent to: one_hot(i) @ embedding_matrix

    Args:
        vocab_size (int): Number of unique tokens in vocabulary
        embed_dim (int): Dimension of embedding vectors (e.g., 64, 128, 256)

    Attributes:
        embeddings (np.ndarray): The embedding matrix of shape (vocab_size, embed_dim)
        grad_embeddings (np.ndarray): Accumulated gradients for embeddings

    Example:
        >>> vocab_size = 20  # 20 unique tokens
        >>> embed_dim = 8    # 8-dimensional embeddings
        >>> embedding = Embedding(vocab_size, embed_dim)
        >>>
        >>> # Token indices for batch
        >>> tokens = np.array([[15, 17, 7], [12, 17, 4]])  # shape: (2, 3)
        >>> embedded = embedding.forward(tokens)  # shape: (2, 3, 8)
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Initialize the embedding layer.

        Initialization Strategy:
            We use small random values (mean=0, std=0.01) to:
            1. Break symmetry - different embeddings start different
            2. Keep values small - prevents exploding gradients early in training
            3. Random initialization - allows the network to learn diverse representations

            Alternative strategies could include:
            - Xavier/Glorot: scale by sqrt(1/vocab_size) for better gradient flow
            - Zeros: would fail due to symmetry (all embeddings would learn the same)
            - Large values: would cause numerical instability

        Args:
            vocab_size (int): Number of unique tokens in vocabulary
            embed_dim (int): Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize embedding matrix with small random values
        # Shape: (vocab_size, embed_dim) - one row per token
        # Each row is the embedding vector for one token
        # Scale factor 0.01 keeps initial values small for stable training
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01

        # Cache for backward pass - stores input token indices
        self.input_indices_cache = None

        # Gradient accumulator - same shape as embeddings
        self.grad_embeddings = None

    def forward(self, token_indices: np.ndarray) -> np.ndarray:
        """
        Forward pass: lookup embeddings for token indices.

        This is a simple array indexing operation, but conceptually it's equivalent
        to multiplying a one-hot encoded matrix by the embedding matrix:
            one_hot(token_indices) @ self.embeddings

        The advantage of direct indexing is efficiency - we don't need to create
        the sparse one-hot matrix.

        Args:
            token_indices (np.ndarray): Integer array of shape (batch_size, seq_len)
                                       Contains token IDs to look up
                                       Each value should be in range [0, vocab_size)

        Returns:
            np.ndarray: Embedded vectors of shape (batch_size, seq_len, embed_dim)
                       Each token ID is replaced with its embedding vector

        Note:
            The same token ID always produces the same embedding vector.
            This is a deterministic lookup, not a learned transformation.
        """
        # Cache input for backward pass
        self.input_indices_cache = token_indices

        # Lookup embeddings - NumPy's fancy indexing
        # This is equivalent to: self.embeddings[token_indices]
        # Result shape: (batch_size, seq_len, embed_dim)
        output = self.embeddings[token_indices]

        return output

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass: accumulate gradients for embedding vectors.

        Gradient Flow:
            The gradient flows back to the embeddings, but NOT to the token indices
            (token indices are discrete and not differentiable).

        Key Challenge - Gradient Accumulation:
            If the same token appears multiple times in the batch, we need to SUM
            all gradients for that token. This is why we iterate and accumulate.

            Example: If token 5 appears at positions [0,1], [2,3], and [5,2],
                    we must add all three gradients: grad[5] = sum of all three

        Why Accumulation Matters:
            Consider the word "the" appearing 10 times in a batch. Each occurrence
            contributes a gradient, and we need to sum them all to properly update
            the embedding for "the".

        Args:
            grad_output (np.ndarray): Gradient from next layer,
                                     shape (batch_size, seq_len, embed_dim)

        Returns:
            None: Token indices are discrete, so gradient doesn't flow back to them
                 The gradient stops here (input is not differentiable)

        Implementation Note:
            This could be optimized using np.add.at() for better performance:
                np.add.at(self.grad_embeddings, token_indices, grad_output)
            But we use explicit loops for clarity in this educational implementation.
        """
        # Initialize gradient accumulator with zeros
        self.grad_embeddings = np.zeros_like(self.embeddings)

        # Get cached token indices
        token_indices = self.input_indices_cache

        # Accumulate gradients for each token
        # We iterate through the batch to handle repeated tokens correctly
        batch_size, seq_len = token_indices.shape

        for i in range(batch_size):
            for j in range(seq_len):
                token_id = token_indices[i, j]
                # Add this gradient to the embedding's gradient
                # If token_id appears multiple times, gradients accumulate
                self.grad_embeddings[token_id] += grad_output[i, j]

        # Note: We don't return grad_input because token indices are discrete
        # and not differentiable. The gradient stops here.
        return None

    def get_parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get all trainable parameters and their gradients.

        Returns:
            list: List of (parameter, gradient) tuples
                 In this case: [(embeddings, grad_embeddings)]

        Note:
            The optimizer will use this to update the embedding matrix during training.
        """
        return [(self.embeddings, self.grad_embeddings)]

    def zero_grad(self) -> None:
        """
        Reset gradients to None.

        This should be called before each backward pass to clear old gradients.
        Otherwise, gradients would accumulate across multiple batches.
        """
        self.grad_embeddings = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
