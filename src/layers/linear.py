"""
Linear (Dense) Layer Module

This module implements a fully connected linear layer from scratch using NumPy.

The linear layer is the fundamental building block of neural networks. It performs
a simple but powerful transformation: y = xW + b, where:
    - x is the input
    - W is the weight matrix
    - b is the bias vector
    - y is the output

Key Concepts:
    - Linear layers learn to transform representations
    - They enable the network to learn complex functions through composition
    - When stacked with non-linearities, they can approximate any function

Why Linear Layers?
    Despite being "just" matrix multiplication, linear layers are powerful because:
    1. They learn meaningful transformations of the input space
    2. When combined with non-linearities (ReLU, etc.), they enable deep learning
    3. They're computationally efficient (matrix multiplication is highly optimized)
    4. They're universal - used in almost every neural network architecture

Implementation Details:
    - Xavier/Glorot initialization for stable gradient flow
    - Support for optional bias term
    - Handles both 2D (batch, features) and 3D (batch, sequence, features) inputs
    - Efficient gradient computation using matrix calculus
"""

import numpy as np
from typing import List, Tuple, Optional


class Linear:
    """
    Fully connected linear layer (also called Dense layer).

    Performs the affine transformation: output = input @ weights + bias

    This is the most fundamental operation in neural networks. Every modern architecture
    from MLPs to Transformers uses linear layers as core components.

    Mathematical Formulation:
        For input x of shape (..., input_dim):
            output = x @ W + b
        where:
            W has shape (input_dim, output_dim)
            b has shape (output_dim,)
            output has shape (..., output_dim)

    The layer learns W and b through backpropagation to minimize the loss.

    Args:
        input_dim (int): Size of input features (number of input neurons)
        output_dim (int): Size of output features (number of output neurons)
        use_bias (bool): Whether to include bias term (default: True)

    Attributes:
        weights (np.ndarray): Weight matrix of shape (input_dim, output_dim)
        bias (np.ndarray): Bias vector of shape (output_dim,) if use_bias=True
        grad_weights (np.ndarray): Gradient of weights
        grad_bias (np.ndarray): Gradient of bias

    Example:
        >>> layer = Linear(input_dim=128, output_dim=64)
        >>> x = np.random.randn(32, 128)  # batch of 32 samples
        >>> output = layer.forward(x)     # shape: (32, 64)
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True) -> None:
        """
        Initialize the linear layer with proper weight initialization.

        Weight Initialization - Xavier/Glorot Initialization:
            We initialize weights uniformly in range [-limit, limit] where:
                limit = sqrt(6 / (input_dim + output_dim))

            Why Xavier Initialization?
            1. Maintains variance of activations across layers
            2. Prevents vanishing/exploding gradients in deep networks
            3. Balances the fan-in (input_dim) and fan-out (output_dim)

            The intuition: If inputs have unit variance, we want outputs to also
            have unit variance. Xavier initialization achieves this for linear
            activations and tanh.

            Alternative Initialization Strategies:
            - He initialization: limit = sqrt(6 / input_dim)
              Better for ReLU activations (accounts for ReLU killing half the neurons)
            - Random normal: weights ~ N(0, 0.01)
              Simple but can cause vanishing/exploding gradients in deep networks
            - Zeros: Would fail completely (all neurons learn the same function)

            Why Uniform Distribution?
            Xavier originally used uniform distribution. Normal distribution
            (with appropriate std) works similarly. We use uniform for:
            - Bounded values (no extreme outliers)
            - Consistent with original paper
            - Slightly faster to generate

        Bias Initialization:
            Bias is initialized to zeros because:
            1. It doesn't affect symmetry breaking (weights do that)
            2. Zero bias is a neutral starting point
            3. The network can easily learn to shift as needed

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
            use_bias (bool): Whether to use bias term
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Xavier/Glorot initialization for weights
        # This scaling ensures variance is preserved through the layer
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))

        if self.use_bias:
            # Initialize bias to zeros (standard practice)
            self.bias = np.zeros(output_dim)
        else:
            self.bias = None

        # Cache for backward pass - stores input for gradient computation
        self.input_cache = None

        # Gradients (computed during backward pass)
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the linear layer.

        Computation:
            output = x @ weights + bias

        This is matrix multiplication followed by bias addition. The @ operator
        performs matrix multiplication, which for our purposes computes:
            output[i,j] = sum_k(x[i,k] * weights[k,j]) + bias[j]

        Why Matrix Multiplication?
            Each output neuron is a weighted sum of all input neurons. Matrix
            multiplication computes all these weighted sums efficiently in parallel.

        Input Shape Flexibility:
            - 2D input (batch_size, input_dim): Standard case
            - 3D input (batch_size, seq_len, input_dim): For sequence models
            The layer automatically handles both by applying the transformation
            to the last dimension.

        Args:
            x (np.ndarray): Input tensor of shape:
                           - (batch_size, input_dim), or
                           - (batch_size, seq_len, input_dim)

        Returns:
            np.ndarray: Output tensor of shape:
                       - (batch_size, output_dim), or
                       - (batch_size, seq_len, output_dim)

        Note:
            The transformation is applied independently to each position in a sequence,
            making it position-independent (important for transformers).
        """
        # Cache input for backward pass
        self.input_cache = x

        # Linear transformation: x @ W
        # np.matmul handles both 2D and 3D inputs automatically
        # For 3D: applies transformation to last dimension at each sequence position
        output = np.matmul(x, self.weights)

        # Add bias if present
        # Broadcasting automatically extends bias across batch and sequence dimensions
        if self.use_bias:
            output = output + self.bias

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the linear layer.

        Computes gradients with respect to inputs, weights, and bias using
        the chain rule of calculus.

        Gradient Derivation:
            Given: L = loss, y = xW + b
            We receive: dL/dy (grad_output)

            Need to compute:
            1. dL/dW = x^T @ (dL/dy)  - gradient for weights
            2. dL/db = sum(dL/dy)     - gradient for bias
            3. dL/dx = (dL/dy) @ W^T  - gradient for input (pass to previous layer)

        Why These Gradients?
            These come from matrix calculus chain rule:
            - Weight gradient: Each weight contributes to every output, so we sum
              over batch and sequence dimensions
            - Bias gradient: Bias is added to every output, so we sum all gradients
            - Input gradient: Each input affects all outputs through different weights,
              so we use weight transpose

        Shape Handling:
            For 3D inputs (batch, seq_len, features), we reshape to 2D for computation,
            then reshape gradients back. This is more efficient than iterating over
            sequence positions.

        Args:
            grad_output (np.ndarray): Gradient from the next layer, same shape as forward output

        Returns:
            np.ndarray: Gradient with respect to input, same shape as forward input
        """
        # Get cached input
        x = self.input_cache

        # Gradient with respect to weights: x^T @ grad_output
        # Need to handle both 2D and 3D inputs
        if x.ndim == 3:
            # Shape: (batch, seq_len, input_dim)
            # Reshape to (batch * seq_len, input_dim) for efficient computation
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.reshape(-1, self.input_dim)
            grad_output_reshaped = grad_output.reshape(-1, self.output_dim)

            # Compute weight gradient
            # Each position in the sequence contributes to the gradient
            self.grad_weights = np.matmul(x_reshaped.T, grad_output_reshaped)

            if self.use_bias:
                # Sum over both batch and sequence dimensions
                # Bias is shared across all positions, so gradients accumulate
                self.grad_bias = np.sum(grad_output_reshaped, axis=0)
        else:
            # Shape: (batch, input_dim)
            self.grad_weights = np.matmul(x.T, grad_output)

            if self.use_bias:
                # Sum over batch dimension only
                self.grad_bias = np.sum(grad_output, axis=0)

        # Gradient with respect to input: grad_output @ W^T
        # This gradient flows back to the previous layer
        grad_input = np.matmul(grad_output, self.weights.T)

        return grad_input

    def get_parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get all trainable parameters and their gradients.

        Returns:
            list: List of (parameter, gradient) tuples
                 [(weights, grad_weights), (bias, grad_bias)] if use_bias=True
                 [(weights, grad_weights)] if use_bias=False

        Note:
            The optimizer uses this to update parameters during training.
        """
        params = [(self.weights, self.grad_weights)]
        if self.use_bias:
            params.append((self.bias, self.grad_bias))
        return params

    def zero_grad(self) -> None:
        """
        Reset gradients to None.

        This should be called before each backward pass to prevent gradient accumulation
        across batches. Without this, gradients would sum up over multiple updates.
        """
        self.grad_weights = None
        self.grad_bias = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Linear(input_dim={self.input_dim}, output_dim={self.output_dim}, use_bias={self.use_bias})"
