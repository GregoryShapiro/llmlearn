"""
Activation Functions Module

This module implements non-linear activation functions from scratch using NumPy.

Why Activation Functions?
    Without activation functions, stacking multiple linear layers would still produce
    a linear function: f(g(x)) = Wx + b is just another linear function.

    Non-linear activations enable neural networks to:
    1. Learn complex, non-linear patterns
    2. Approximate any continuous function (universal approximation theorem)
    3. Create hierarchical representations

Key Activation Functions:
    - ReLU: Fast, effective, prevents vanishing gradients (but has dying ReLU problem)
    - Softmax: Converts logits to probabilities (used for classification)

Alternative Activations (not implemented here):
    - Sigmoid: σ(x) = 1/(1 + e^(-x)) - squashes to [0,1], vanishing gradient problem
    - Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) - squashes to [-1,1]
    - GELU: Gaussian Error Linear Unit - used in BERT and GPT
    - Swish/SiLU: x * σ(x) - smooth, performs well in deep networks
"""

import numpy as np

from typing import List, Tuple, Optional

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.

    Applies the function: ReLU(x) = max(0, x)

    ReLU is the most popular activation function in deep learning because:
    1. Computationally efficient (just a comparison and selection)
    2. Doesn't saturate for positive values (no vanishing gradient)
    3. Sparse activation (many neurons output 0)
    4. Empirically works very well in practice

    Why ReLU Over Sigmoid/Tanh?
        Sigmoid and tanh saturate (flatten out) for large positive/negative values,
        causing vanishing gradients. ReLU doesn't saturate for positive values,
        allowing gradients to flow freely.

    The Dying ReLU Problem:
        If a neuron gets a large negative gradient, it might never activate again
        (always outputs 0). Solutions include:
        - Leaky ReLU: max(αx, x) with small α like 0.01
        - ELU: α(e^x - 1) for x < 0
        - We use standard ReLU for simplicity

    Mathematical Properties:
        - Non-linear: Enables learning complex patterns
        - Non-differentiable at x=0: We treat gradient as 0 at this point
        - Piecewise linear: Easy to optimize
        - Unbounded above: Can produce arbitrarily large outputs

    Args:
        None - ReLU has no parameters

    Example:
        >>> relu = ReLU()
        >>> x = np.array([-2, -1, 0, 1, 2])
        >>> output = relu.forward(x)  # [0, 0, 0, 1, 2]
    """

    def __init__(self) -> None:
        """
        Initialize ReLU activation.

        No parameters to initialize since ReLU is a fixed function.
        """
        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ReLU.

        Applies element-wise maximum between 0 and input.

        Computation:
            output[i] = max(0, x[i]) for each element i

        Implementation:
            We use np.maximum which is highly optimized and vectorized.
            This is equivalent to: np.where(x > 0, x, 0)

        Why Cache Input?
            We need to know which values were positive for the backward pass.
            When x > 0, gradient is 1. When x <= 0, gradient is 0.

        Args:
            x (np.ndarray): Input tensor of any shape

        Returns:
            np.ndarray: Output tensor of same shape with negative values zeroed
        """
        # Cache input for backward pass (need to know which values were positive)
        self.input_cache = x

        # Apply ReLU: max(0, x) element-wise
        # np.maximum is vectorized and efficient
        output = np.maximum(0, x)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through ReLU.

        Gradient Rule:
            - If input > 0: gradient = 1 (pass through)
            - If input <= 0: gradient = 0 (block)

        This makes sense intuitively:
            - For positive inputs, ReLU is just identity (y = x), so dy/dx = 1
            - For negative inputs, ReLU is constant (y = 0), so dy/dx = 0

        Implementation:
            We create a binary mask where True indicates input was positive.
            Multiplying by this mask zeros out gradients for negative inputs.

        Mathematical Note:
            Technically, ReLU is not differentiable at x=0. In practice, we define
            the gradient as 0 at this point (though 1 also works). This choice
            rarely matters because the probability of exactly hitting 0 is negligible.

        Args:
            grad_output (np.ndarray): Gradient from next layer, same shape as forward output

        Returns:
            np.ndarray: Gradient with respect to input (zero where input was negative)
        """
        # Gradient is 1 where input > 0, else 0
        # The > comparison creates a boolean array, which when multiplied acts as a mask
        grad_input = grad_output * (self.input_cache > 0)

        return grad_input

    def get_parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get trainable parameters.

        Returns:
            list: Empty list (ReLU has no parameters)
        """
        return []

    def zero_grad(self) -> None:
        """No gradients to reset (no parameters)."""
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return "ReLU()"


class Softmax:
    """
    Softmax activation function.

    Converts a vector of real numbers (logits) into a probability distribution.

    Formula:
        softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

    Key Properties:
        1. Output sums to 1 (valid probability distribution)
        2. All outputs are positive (in range [0, 1])
        3. Preserves ordering (larger input = larger output)
        4. Differentiable everywhere

    Why Softmax?
        For classification, we need to convert raw scores (logits) into probabilities.
        Softmax does this while:
        - Emphasizing the largest values (exponential amplification)
        - Ensuring valid probabilities (sum to 1)
        - Being differentiable (enables backpropagation)

    Numerical Stability:
        Naively computing exp(x) can overflow for large x or underflow for negative x.
        We use the mathematical identity:
            softmax(x) = softmax(x - max(x))
        This shifts all values down but produces the same result, preventing overflow.

    Temperature Parameter (not implemented):
        softmax(x/T) with temperature T:
        - T → 0: Approaches one-hot (winner-take-all)
        - T → ∞: Approaches uniform distribution
        - T = 1: Standard softmax (our implementation)

    Args:
        axis (int): Axis along which to apply softmax (default: -1, last dimension)

    Example:
        >>> softmax = Softmax()
        >>> logits = np.array([[1.0, 2.0, 3.0]])
        >>> probs = softmax.forward(logits)  # [[0.09, 0.24, 0.67]]
        >>> # Largest logit (3.0) gets highest probability (0.67)
    """

    def __init__(self, axis: int = -1) -> None:
        """
        Initialize Softmax.

        Args:
            axis (int): Axis along which to apply softmax
                       -1 means last axis (typical for classification)
        """
        self.axis = axis
        self.output_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through softmax.

        Implementation uses numerical stability trick: subtract max before exp.

        Why Subtract Max?
            Without it, exp(large_number) can overflow to infinity.
            With it, we compute exp(small_number) which is safe.

            Proof that it's equivalent:
                softmax(x_i) = exp(x_i) / sum(exp(x_j))
                             = exp(x_i - c) / sum(exp(x_j - c))  [for any constant c]

            Choosing c = max(x) ensures all exponents are <= 0, preventing overflow.

        Args:
            x (np.ndarray): Input tensor (logits)
                           Common shape: (batch_size, num_classes)

        Returns:
            np.ndarray: Probability distribution of same shape
                       Values in [0, 1] and sum to 1 along specified axis

        Example:
            >>> x = np.array([[1, 2, 3], [1, 2, 3]])
            >>> softmax = Softmax(axis=-1)
            >>> probs = softmax.forward(x)
            >>> np.sum(probs, axis=-1)  # [1., 1.] - sums to 1 per row
        """
        # Numerical stability: subtract max to prevent overflow
        # keepdims=True ensures max has compatible shape for broadcasting
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)

        # Normalize to get probabilities (sum = 1)
        output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        # Cache for backward pass - we'll need softmax outputs
        self.output_cache = output

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through softmax.

        The gradient of softmax is more complex than other activations because
        each output depends on all inputs (through the normalization).

        Jacobian of Softmax:
            For softmax output s_i:
                ∂s_i/∂x_j = s_i * (δ_ij - s_j)

            Where δ_ij is the Kronecker delta (1 if i==j, else 0).

        Intuition:
            - When i==j (diagonal): s_i(1 - s_i) - output scaled by (1 - output)
            - When i!=j (off-diagonal): -s_i*s_j - negative interaction term

        Simplified Gradient Formula:
            grad_input_i = sum_j(grad_output_j * ∂s_j/∂x_i)
                         = s_i * (grad_output_i - sum_j(grad_output_j * s_j))

        This can be rewritten as:
            grad_input = s * (grad_output - sum(grad_output * s))

        Where * is element-wise multiplication.

        Why This Formula?
            It captures how changing input x_i affects:
            1. Its own probability s_i (direct effect)
            2. All other probabilities s_j (indirect through normalization)

        Args:
            grad_output (np.ndarray): Gradient from next layer

        Returns:
            np.ndarray: Gradient with respect to input (logits)
        """
        # Get cached softmax output
        s = self.output_cache

        # Compute gradient using the simplified formula
        # Step 1: Compute sum of (grad_output * softmax) along the softmax axis
        sum_term = np.sum(grad_output * s, axis=self.axis, keepdims=True)

        # Step 2: Compute grad_input = s * (grad_output - sum_term)
        # This implements the Jacobian-vector product efficiently
        grad_input = s * (grad_output - sum_term)

        return grad_input

    def get_parameters(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get trainable parameters.

        Returns:
            list: Empty list (Softmax has no parameters)
        """
        return []

    def zero_grad(self) -> None:
        """No gradients to reset (no parameters)."""
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Softmax(axis={self.axis})"
