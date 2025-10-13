"""
Layer Normalization Module

This module implements Layer Normalization, a technique for stabilizing training
in deep neural networks.

The Problem - Internal Covariate Shift:
    As we train deep networks, the distribution of inputs to each layer changes
    as previous layers' parameters update. This makes training unstable and slow.

The Solution - Normalization:
    Normalize activations to have consistent statistics (mean=0, std=1).
    This stabilizes training and allows higher learning rates.

Why Layer Normalization (vs Batch Normalization)?
    Batch Normalization normalizes across the batch dimension.
    Layer Normalization normalizes across the feature dimension.

    For Transformers, Layer Norm is preferred because:
    1. Works with any batch size (including batch_size=1)
    2. No dependence on batch statistics (important for sequence models)
    3. Works well with variable-length sequences
    4. Normalizes each token independently (good for attention mechanisms)

Mathematical Formulation:
    Given input x of shape (..., features):
    1. Compute mean: μ = mean(x) across features
    2. Compute variance: σ² = var(x) across features
    3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
    4. Scale and shift: y = γ * x_norm + β

    Where:
    - ε (epsilon) prevents division by zero
    - γ (gamma) is learned scale parameter
    - β (beta) is learned shift parameter

Why Learnable Parameters (γ and β)?
    After normalization, all features have mean=0 and std=1. This might be too
    restrictive! The learnable parameters allow the network to:
    - Undo normalization if beneficial (γ=σ, β=μ recovers original distribution)
    - Learn optimal scale and shift for each feature
    - Maintain representational power

Implementation Note:
    This is a simplified but correct implementation. Production implementations
    (PyTorch, TensorFlow) include optimizations and handle edge cases more carefully.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class LayerNorm:
    """
    Layer Normalization.

    Normalizes activations across the feature dimension to have mean=0 and variance=1,
    then applies learned scale (gamma) and shift (beta) parameters.

    This helps stabilize training in deep networks by ensuring consistent activation
    distributions. It's particularly important in transformers, appearing after every
    attention and feed-forward sub-layer.

    Args:
        normalized_shape (int): Dimension to normalize (typically embed_dim)
        eps (float): Small constant for numerical stability (default: 1e-5)

    Attributes:
        gamma (np.ndarray): Learnable scale parameter, shape (normalized_shape,)
        beta (np.ndarray): Learnable shift parameter, shape (normalized_shape,)
        grad_gamma (np.ndarray): Gradient of gamma
        grad_beta (np.ndarray): Gradient of beta

    Example:
        >>> layer_norm = LayerNorm(normalized_shape=64)
        >>> x = np.random.randn(32, 10, 64)  # (batch, seq_len, features)
        >>> normalized = layer_norm.forward(x)
        >>> # Each token is normalized independently across its 64 features
    """

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5) -> None:
        """
        Initialize layer normalization.

        Initialization Strategy:
            - gamma (scale) initialized to ones: y = 1 * x_norm + 0 = x_norm
            - beta (shift) initialized to zeros: starts as pure normalization
            - This is the identity transformation on normalized values
            - The network can learn to adjust if needed

        Why This Initialization?
            Starting with identity transformation on normalized values ensures:
            1. Training starts with reasonable activations (mean=0, std=1)
            2. Network can gradually learn to adjust scale/shift if beneficial
            3. If normalization helps as-is, parameters can stay near initialization

        Args:
            normalized_shape (int): Number of features to normalize
            eps (float): Epsilon for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters - initialized to implement identity on normalized values
        self.gamma = np.ones(normalized_shape)   # Scale parameter
        self.beta = np.zeros(normalized_shape)   # Shift parameter

        # Cache for backward pass
        self.input_cache = None
        self.normalized_cache = None
        self.std_cache = None

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through layer normalization.

        Normalization Process:
            1. Compute statistics (mean, variance) per feature
            2. Normalize to mean=0, variance=1
            3. Scale by gamma and shift by beta

        Why Normalize Per Token?
            In transformers, each token's representation should be independently
            normalized. This ensures:
            - Token representations have consistent scale
            - Attention weights are stable
            - Gradients flow smoothly

        Numerical Stability:
            We add eps (1e-5) to variance before taking square root to prevent:
            - Division by zero if variance is zero
            - Numerical instability from very small variances

        Args:
            x (np.ndarray): Input tensor of shape (..., normalized_shape)
                           Common shapes:
                           - (batch, features) for MLP
                           - (batch, seq_len, features) for transformers

        Returns:
            np.ndarray: Normalized, scaled, and shifted tensor of same shape
        """
        # Cache input for backward pass
        self.input_cache = x

        # Compute mean and variance across the last dimension (features)
        # keepdims=True ensures mean has shape (..., 1) for broadcasting
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize: (x - mean) / sqrt(variance + eps)
        # The eps term prevents division by zero and improves stability
        std = np.sqrt(variance + self.eps)
        normalized = (x - mean) / std

        # Cache for backward pass
        self.normalized_cache = normalized
        self.std_cache = std

        # Scale and shift: gamma * normalized + beta
        # Broadcasting applies the same gamma/beta to all batch and sequence positions
        output = self.gamma * normalized + self.beta

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through layer normalization.

        This is one of the more complex backward passes because normalization
        couples all features together (changing one feature affects the mean and
        variance, which affects all other features).

        Gradient Components:
            1. grad_gamma: How much does loss change if we scale differently?
            2. grad_beta: How much does loss change if we shift differently?
            3. grad_input: How much does loss change if we change the input?

        Why This Is Complex:
            The mean and variance depend on all input features, so the gradient
            of normalization with respect to input involves all features.
            We must account for:
            - Direct gradient through normalization
            - Indirect gradient through mean
            - Indirect gradient through variance

        Mathematical Derivation:
            Given y = γ * ((x - μ) / σ) + β, we need ∂L/∂x

            Let x̂ = (x - μ) / σ (normalized value)

            Chain rule gives us:
            ∂L/∂x = ∂L/∂x̂ * ∂x̂/∂x + ∂L/∂μ * ∂μ/∂x + ∂L/∂σ * ∂σ/∂x

            Where:
            - ∂L/∂x̂ = grad_output * γ
            - ∂μ/∂x = 1/N
            - ∂σ/∂x involves (x - μ)

        Args:
            grad_output (np.ndarray): Gradient from next layer

        Returns:
            np.ndarray: Gradient with respect to input

        Implementation Note:
            This implements the gradient carefully to handle the coupling between
            features. The formulas look complex but follow directly from calculus.
        """
        # Get cached values
        x = self.input_cache
        normalized = self.normalized_cache
        std = self.std_cache

        # Gradients for learnable parameters (gamma and beta)
        # grad_gamma: sum over all positions where gamma is applied
        # grad_beta: sum over all positions where beta is applied
        # We sum over all dimensions except the last (feature dimension)
        self.grad_gamma = np.sum(grad_output * normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient for input (more complex due to mean and variance coupling)
        N = x.shape[-1]  # Number of features being normalized

        # Gradient of normalized values
        grad_normalized = grad_output * self.gamma

        # Gradient through normalization (involves mean and variance)
        # This is the tricky part - the gradient flows through three paths:
        # 1. Direct path through normalization
        # 2. Indirect path through mean
        # 3. Indirect path through variance

        # Compute gradient through variance
        # variance = E[(x - mean)^2], so ∂variance/∂x involves (x - mean)
        grad_variance = np.sum(grad_normalized * (x - np.mean(x, axis=-1, keepdims=True)),
                               axis=-1, keepdims=True)
        grad_variance *= -0.5 * np.power(std, -3)

        # Compute gradient through mean
        grad_mean = np.sum(grad_normalized * -1.0 / std, axis=-1, keepdims=True)
        grad_mean += grad_variance * np.mean(-2.0 * (x - np.mean(x, axis=-1, keepdims=True)),
                                             axis=-1, keepdims=True)

        # Combine all gradient paths
        grad_input = grad_normalized / std
        grad_input += grad_variance * 2.0 * (x - np.mean(x, axis=-1, keepdims=True)) / N
        grad_input += grad_mean / N

        return grad_input

    def get_parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get trainable parameters and their gradients.

        Returns:
            list: [(gamma, grad_gamma), (beta, grad_beta)]

        Note:
            These parameters are updated by the optimizer during training.
        """
        return [
            (self.gamma, self.grad_gamma),
            (self.beta, self.grad_beta)
        ]

    def zero_grad(self) -> None:
        """
        Reset gradients to None.

        Called before each backward pass to clear previous gradients.
        """
        self.grad_gamma = None
        self.grad_beta = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
