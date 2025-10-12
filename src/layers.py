"""
Neural Network Layers Module

This module implements basic neural network layers from scratch using NumPy.
All layers support both forward and backward passes for training.
"""

import numpy as np


class Embedding:
    """
    Embedding layer that converts token indices to dense vectors.
    
    This is essentially a lookup table where each token ID maps to a learned vector.
    For example, token 15 (Max) → [0.2, -0.5, 0.8, ..., 0.1]
    """
    
    def __init__(self, vocab_size, embed_dim):
        """
        Initialize the embedding layer.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary
            embed_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embedding matrix
        # Shape: (vocab_size, embed_dim)
        # Each row is the embedding vector for one token
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # Cache for backward pass
        self.input_indices_cache = None
        
        # Gradient
        self.grad_embeddings = None
    
    def forward(self, token_indices):
        """
        Forward pass: lookup embeddings for token indices.
        
        Args:
            token_indices: Integer array of shape (batch_size, seq_len)
                          Contains token IDs to look up
        
        Returns:
            Embedded vectors of shape (batch_size, seq_len, embed_dim)
        """
        # Cache input for backward pass
        self.input_indices_cache = token_indices
        
        # Lookup embeddings
        # This is equivalent to one-hot encoding followed by matrix multiplication
        output = self.embeddings[token_indices]
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass: accumulate gradients for embedding vectors.
        
        Args:
            grad_output: Gradient from next layer, shape (batch_size, seq_len, embed_dim)
        
        Returns:
            None (token indices are not differentiable)
        """
        # Initialize gradient accumulator
        self.grad_embeddings = np.zeros_like(self.embeddings)
        
        # Get cached indices
        token_indices = self.input_indices_cache
        
        # Accumulate gradients for each token
        # Multiple positions might use the same token, so we need to sum their gradients
        batch_size, seq_len = token_indices.shape
        
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = token_indices[i, j]
                self.grad_embeddings[token_id] += grad_output[i, j]
        
        # Note: We don't return grad_input because token indices are discrete
        # and not differentiable. The gradient stops here.
        return None
    
    def get_parameters(self):
        """
        Get all trainable parameters.
        
        Returns:
            List of (parameter, gradient) tuples
        """
        return [(self.embeddings, self.grad_embeddings)]
    
    def zero_grad(self):
        """Reset gradients to None."""
        self.grad_embeddings = None
    
    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class Linear:
    """
    Fully connected linear layer (also called Dense layer).
    
    Performs the operation: output = input @ weights + bias
    
    This is the fundamental building block of neural networks.
    """
    
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        Initialize the linear layer.
        
        Args:
            input_dim: Size of input features
            output_dim: Size of output features
            use_bias: Whether to include bias term (default: True)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Initialize weights using Xavier/Glorot initialization
        # This helps with gradient flow in deep networks
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        
        if self.use_bias:
            # Initialize bias to zeros
            self.bias = np.zeros(output_dim)
        else:
            self.bias = None
        
        # Cache for backward pass
        self.input_cache = None
        
        # Gradients (will be computed during backward pass)
        self.grad_weights = None
        self.grad_bias = None
    
    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim) or (batch_size, seq_len, output_dim)
        """
        # Cache input for backward pass
        self.input_cache = x
        
        # Linear transformation: x @ W
        output = np.matmul(x, self.weights)
        
        # Add bias if present
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through the linear layer.
        
        Computes gradients with respect to inputs, weights, and bias.
        
        Args:
            grad_output: Gradient from the next layer, same shape as forward output
        
        Returns:
            grad_input: Gradient with respect to input, same shape as forward input
        """
        # Get cached input
        x = self.input_cache
        
        # Gradient with respect to weights: x^T @ grad_output
        # Need to handle both 2D and 3D inputs
        if x.ndim == 3:
            # Shape: (batch, seq_len, input_dim)
            # Reshape to (batch * seq_len, input_dim) for easier computation
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.reshape(-1, self.input_dim)
            grad_output_reshaped = grad_output.reshape(-1, self.output_dim)
            
            self.grad_weights = np.matmul(x_reshaped.T, grad_output_reshaped)
            
            if self.use_bias:
                # Sum over batch and sequence dimensions
                self.grad_bias = np.sum(grad_output_reshaped, axis=0)
        else:
            # Shape: (batch, input_dim)
            self.grad_weights = np.matmul(x.T, grad_output)
            
            if self.use_bias:
                # Sum over batch dimension
                self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient with respect to input: grad_output @ W^T
        grad_input = np.matmul(grad_output, self.weights.T)
        
        return grad_input
    
    def get_parameters(self):
        """
        Get all trainable parameters.
        
        Returns:
            List of (parameter, gradient) tuples
        """
        params = [(self.weights, self.grad_weights)]
        if self.use_bias:
            params.append((self.bias, self.grad_bias))
        return params
    
    def zero_grad(self):
        """Reset gradients to None."""
        self.grad_weights = None
        self.grad_bias = None
    
    def __repr__(self):
        return f"Linear(input_dim={self.input_dim}, output_dim={self.output_dim}, use_bias={self.use_bias})"


class PositionalEncoding:
    """
    Positional Encoding using sinusoidal functions.

    Adds position information to embeddings so the model knows the order of tokens.
    Uses fixed sinusoidal encoding (not learned), which generalizes to any sequence length.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_seq_len, embed_dim):
        """
        Initialize positional encoding.

        Args:
            max_seq_len: Maximum sequence length to precompute encodings for
            embed_dim: Dimension of embeddings (must match embedding layer)
        """
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Precompute positional encodings
        # Shape: (max_seq_len, embed_dim)
        self.encodings = self._create_positional_encodings()

    def _create_positional_encodings(self):
        """
        Create sinusoidal positional encodings.

        Returns:
            Positional encoding matrix of shape (max_seq_len, embed_dim)
        """
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        position = np.arange(self.max_seq_len)[:, np.newaxis]  # Shape: (max_seq_len, 1)

        # Create dimension indices: [0, 2, 4, ..., embed_dim-2] (even indices)
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))

        # Initialize encoding matrix
        encodings = np.zeros((self.max_seq_len, self.embed_dim))

        # Apply sin to even indices
        encodings[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd indices
        encodings[:, 1::2] = np.cos(position * div_term)

        return encodings

    def forward(self, embedded_tokens):
        """
        Add positional encodings to embedded tokens.

        Args:
            embedded_tokens: Tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor with positional encodings added, same shape as input
        """
        batch_size, seq_len, embed_dim = embedded_tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        if embed_dim != self.embed_dim:
            raise ValueError(f"Embedding dimension {embed_dim} does not match {self.embed_dim}")

        # Get encodings for this sequence length
        # Shape: (seq_len, embed_dim)
        pos_encodings = self.encodings[:seq_len, :]

        # Add to embedded tokens (broadcasting across batch dimension)
        output = embedded_tokens + pos_encodings

        return output

    def backward(self, grad_output):
        """
        Backward pass for positional encoding.

        Since positional encodings are fixed (not learned), the gradient
        just flows through unchanged.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Same gradient (positional encoding has no learnable parameters)
        """
        # Gradient flows through unchanged
        return grad_output

    def get_parameters(self):
        """
        Get trainable parameters.

        Returns:
            Empty list (positional encodings are fixed, not learned)
        """
        return []

    def zero_grad(self):
        """No gradients to reset (no learnable parameters)."""
        pass

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"


class LayerNorm:
    """
    Layer Normalization.

    Normalizes activations across the feature dimension to have mean=0 and variance=1,
    then applies learned scale (gamma) and shift (beta) parameters.

    This helps stabilize training in deep networks.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize layer normalization.

        Args:
            normalized_shape: Dimension to normalize (typically embed_dim)
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(normalized_shape)  # Scale
        self.beta = np.zeros(normalized_shape)  # Shift

        # Cache for backward pass
        self.input_cache = None
        self.normalized_cache = None
        self.std_cache = None

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        """
        Forward pass through layer normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized, scaled, and shifted tensor of same shape
        """
        # Cache input for backward pass
        self.input_cache = x

        # Compute mean and variance across the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        std = np.sqrt(variance + self.eps)
        normalized = (x - mean) / std

        # Cache for backward pass
        self.normalized_cache = normalized
        self.std_cache = std

        # Scale and shift
        output = self.gamma * normalized + self.beta

        return output

    def backward(self, grad_output):
        """
        Backward pass through layer normalization.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Get cached values
        x = self.input_cache
        normalized = self.normalized_cache
        std = self.std_cache

        # Gradients for gamma and beta
        self.grad_gamma = np.sum(grad_output * normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient for input (more complex due to mean and variance)
        N = x.shape[-1]

        # Gradient of normalized values
        grad_normalized = grad_output * self.gamma

        # Gradient through normalization
        # This is the gradient of (x - mean) / std with respect to x
        grad_variance = np.sum(grad_normalized * (x - np.mean(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
        grad_variance *= -0.5 * np.power(std, -3)

        grad_mean = np.sum(grad_normalized * -1.0 / std, axis=-1, keepdims=True)
        grad_mean += grad_variance * np.mean(-2.0 * (x - np.mean(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)

        grad_input = grad_normalized / std
        grad_input += grad_variance * 2.0 * (x - np.mean(x, axis=-1, keepdims=True)) / N
        grad_input += grad_mean / N

        return grad_input

    def get_parameters(self):
        """
        Get trainable parameters.

        Returns:
            List of (parameter, gradient) tuples
        """
        return [
            (self.gamma, self.grad_gamma),
            (self.beta, self.grad_beta)
        ]

    def zero_grad(self):
        """Reset gradients to None."""
        self.grad_gamma = None
        self.grad_beta = None

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class ReLU:
    """
    Rectified Linear Unit activation function.

    Applies the function: ReLU(x) = max(0, x)

    This is the most common activation function in neural networks.
    """

    def __init__(self):
        """Initialize ReLU activation."""
        self.input_cache = None

    def forward(self, x):
        """
        Forward pass through ReLU.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of same shape with negative values zeroed
        """
        # Cache input for backward pass
        self.input_cache = x

        # Apply ReLU: max(0, x)
        output = np.maximum(0, x)

        return output

    def backward(self, grad_output):
        """
        Backward pass through ReLU.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input (zero where input was negative)
        """
        # Gradient is 1 where input > 0, else 0
        grad_input = grad_output * (self.input_cache > 0)

        return grad_input

    def get_parameters(self):
        """
        Get trainable parameters.

        Returns:
            Empty list (ReLU has no parameters)
        """
        return []

    def zero_grad(self):
        """No gradients to reset (no parameters)."""
        pass

    def __repr__(self):
        return "ReLU()"


class Softmax:
    """
    Softmax activation function.

    Converts a vector of values into a probability distribution.
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Used in the final layer for classification.
    """

    def __init__(self, axis=-1):
        """
        Initialize Softmax.

        Args:
            axis: Axis along which to apply softmax (default: -1, last dimension)
        """
        self.axis = axis
        self.output_cache = None

    def forward(self, x):
        """
        Forward pass through softmax.

        Uses numerical stability trick: subtract max before exp.

        Args:
            x: Input tensor (logits)

        Returns:
            Probability distribution (same shape as input, sums to 1 along axis)
        """
        # Numerical stability: subtract max
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)

        # Normalize to get probabilities
        output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        # Cache for backward pass
        self.output_cache = output

        return output

    def backward(self, grad_output):
        """
        Backward pass through softmax.

        The Jacobian of softmax is:
            J_ij = softmax_i * (delta_ij - softmax_j)

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Get cached softmax output
        s = self.output_cache

        # Compute gradient
        # grad_input_i = sum_j(grad_output_j * J_ij)
        #              = sum_j(grad_output_j * s_i * (delta_ij - s_j))
        #              = s_i * (grad_output_i - sum_j(grad_output_j * s_j))

        # Sum of grad_output * softmax along axis
        sum_term = np.sum(grad_output * s, axis=self.axis, keepdims=True)

        # Compute gradient
        grad_input = s * (grad_output - sum_term)

        return grad_input

    def get_parameters(self):
        """
        Get trainable parameters.

        Returns:
            Empty list (Softmax has no parameters)
        """
        return []

    def zero_grad(self):
        """No gradients to reset (no parameters)."""
        pass

    def __repr__(self):
        return f"Softmax(axis={self.axis})"


# ============================================================================
# Testing and demonstration
# ============================================================================

def test_embedding_basic():
    """Test basic embedding lookup."""
    print("Test 1: Basic Embedding Lookup")
    print("-" * 60)
    
    # Create embedding layer
    vocab_size = 20
    embed_dim = 8
    embedding = Embedding(vocab_size, embed_dim)
    
    print(f"Embedding: {embedding}")
    print(f"Embedding matrix shape: {embedding.embeddings.shape}")
    print()
    
    # Create token indices
    batch_size = 2
    seq_len = 5
    token_indices = np.array([
        [15, 17, 7, 19, 5],   # First sequence
        [12, 17, 4, 19, 10],  # Second sequence
    ])
    
    print(f"Token indices shape: {token_indices.shape}")
    print(f"Token indices:\n{token_indices}")
    print()
    
    # Forward pass
    embedded = embedding.forward(token_indices)
    
    print(f"Embedded output shape: {embedded.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {embed_dim})")
    print(f"Shapes match: {embedded.shape == (batch_size, seq_len, embed_dim)}")
    print()
    
    # Check that same token IDs produce same embeddings
    print(f"Token at position [0,1]: {token_indices[0,1]} → embedding[0:3]: {embedded[0,1,:3]}")
    print(f"Token at position [1,1]: {token_indices[1,1]} → embedding[0:3]: {embedded[1,1,:3]}")
    print(f"Same token ID (17), same embedding: {np.allclose(embedded[0,1], embedded[1,1])}")
    print()


def test_embedding_gradients():
    """Test embedding gradient accumulation."""
    print("Test 2: Embedding Gradient Accumulation")
    print("-" * 60)
    
    # Small embedding for easy verification
    vocab_size = 5
    embed_dim = 3
    embedding = Embedding(vocab_size, embed_dim)
    
    # Token indices
    token_indices = np.array([
        [0, 1, 2],
        [1, 3, 4],
    ])
    
    print(f"Token indices:\n{token_indices}")
    print()
    
    # Forward pass
    embedded = embedding.forward(token_indices)
    
    # Create gradient (all ones for simplicity)
    grad_output = np.ones_like(embedded)
    
    print(f"Gradient output shape: {grad_output.shape}")
    print(f"Gradient output (all ones):\n{grad_output[0]}")
    print()
    
    # Backward pass
    embedding.backward(grad_output)
    
    print(f"Gradient embeddings shape: {embedding.grad_embeddings.shape}")
    print(f"Gradient embeddings:\n{embedding.grad_embeddings}")
    print()
    
    # Verify gradient accumulation
    # Token 1 appears twice (positions [0,1] and [1,0]), so its gradient should be 2x
    print("Gradient analysis:")
    for token_id in range(vocab_size):
        count = np.sum(token_indices == token_id)
        expected_grad_sum = count * embed_dim  # Since grad is all ones
        actual_grad_sum = np.sum(embedding.grad_embeddings[token_id])
        print(f"  Token {token_id}: appears {count} times, grad sum = {actual_grad_sum:.1f} (expected {expected_grad_sum:.1f})")
    print()


def test_embedding_same_token():
    """Test that repeated tokens accumulate gradients correctly."""
    print("Test 3: Repeated Token Gradient Accumulation")
    print("-" * 60)
    
    vocab_size = 10
    embed_dim = 4
    embedding = Embedding(vocab_size, embed_dim)
    
    # Same token repeated multiple times
    token_indices = np.array([
        [5, 5, 5, 5],
        [5, 5, 5, 5],
    ])
    
    print(f"Token indices (all token 5):\n{token_indices}")
    print()
    
    # Forward pass
    embedded = embedding.forward(token_indices)
    
    # Gradient: all ones
    grad_output = np.ones_like(embedded)
    
    # Backward pass
    embedding.backward(grad_output)
    
    print(f"Gradient for token 5:\n{embedding.grad_embeddings[5]}")
    print(f"Expected (8 occurrences × 1.0 gradient): all {8.0}s")
    print(f"Actual values match expected: {np.allclose(embedding.grad_embeddings[5], 8.0)}")
    print()
    
    # Other tokens should have zero gradient
    print("Gradients for other tokens (should be zero):")
    for token_id in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
        grad_sum = np.sum(np.abs(embedding.grad_embeddings[token_id]))
        print(f"  Token {token_id}: {grad_sum:.6f}")
    print()


def test_embedding_with_padding():
    """Test embedding with padding tokens."""
    print("Test 4: Embedding with Padding Tokens")
    print("-" * 60)
    
    vocab_size = 20
    embed_dim = 8
    embedding = Embedding(vocab_size, embed_dim)
    
    PAD_TOKEN = 0
    
    # Sequences with padding (token 0)
    token_indices = np.array([
        [15, 17, 7, 19, 5, 0, 0, 0],   # 5 real tokens, 3 padding
        [12, 17, 4, 0, 0, 0, 0, 0],    # 3 real tokens, 5 padding
    ])
    
    print(f"Token indices (0 = padding):\n{token_indices}")
    print()
    
    # Forward pass
    embedded = embedding.forward(token_indices)
    
    print(f"Embedded shape: {embedded.shape}")
    print(f"Embedding for padding token (first 4 dims): {embedded[0, 5, :4]}")
    print(f"(Padding tokens get embedded too, but will be masked in attention)")
    print()
    
    # Backward with gradient
    grad_output = np.random.randn(*embedded.shape)
    embedding.backward(grad_output)
    
    # Check that padding token accumulated gradients
    padding_grad_sum = np.sum(np.abs(embedding.grad_embeddings[PAD_TOKEN]))
    print(f"Gradient accumulated for padding token: {padding_grad_sum:.4f}")
    print("(In practice, attention masks prevent padding from affecting the loss)")
    print()


def test_linear_2d():
    """Test Linear layer with 2D input (batch_size, input_dim)."""
    print("Test 1: Linear Layer with 2D Input")
    print("-" * 60)
    
    # Create layer
    layer = Linear(input_dim=4, output_dim=3)
    print(f"Layer: {layer}")
    print(f"Weights shape: {layer.weights.shape}")
    print(f"Bias shape: {layer.bias.shape}")
    print()
    
    # Forward pass
    batch_size = 2
    x = np.random.randn(batch_size, 4)
    print(f"Input shape: {x.shape}")
    
    output = layer.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    print()
    
    # Backward pass
    grad_output = np.random.randn(batch_size, 3)
    print(f"Gradient output shape: {grad_output.shape}")
    
    grad_input = layer.backward(grad_output)
    print(f"Gradient input shape: {grad_input.shape}")
    print(f"Gradient weights shape: {layer.grad_weights.shape}")
    print(f"Gradient bias shape: {layer.grad_bias.shape}")
    print()


def test_linear_3d():
    """Test Linear layer with 3D input (batch_size, seq_len, input_dim)."""
    print("Test 2: Linear Layer with 3D Input (Sequence)")
    print("-" * 60)
    
    # Create layer
    layer = Linear(input_dim=5, output_dim=3)
    print(f"Layer: {layer}")
    print()
    
    # Forward pass with sequence
    batch_size = 2
    seq_len = 4
    x = np.random.randn(batch_size, seq_len, 5)
    print(f"Input shape: {x.shape} (batch, seq_len, input_dim)")
    
    output = layer.forward(x)
    print(f"Output shape: {output.shape} (batch, seq_len, output_dim)")
    print()
    
    # Backward pass
    grad_output = np.random.randn(batch_size, seq_len, 3)
    grad_input = layer.backward(grad_output)
    
    print(f"Gradient input shape: {grad_input.shape}")
    print(f"Gradient weights shape: {layer.grad_weights.shape}")
    print(f"Gradient bias shape: {layer.grad_bias.shape}")
    print()


def test_linear_no_bias():
    """Test Linear layer without bias."""
    print("Test 3: Linear Layer without Bias")
    print("-" * 60)
    
    layer = Linear(input_dim=3, output_dim=2, use_bias=False)
    print(f"Layer: {layer}")
    print(f"Has bias: {layer.use_bias}")
    print()
    
    x = np.random.randn(2, 3)
    output = layer.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def test_gradient_numerical():
    """Test gradients using numerical approximation."""
    print("Test 4: Numerical Gradient Check")
    print("-" * 60)
    
    # Small layer for easier verification
    layer = Linear(input_dim=2, output_dim=2)
    
    # Simple input
    x = np.array([[1.0, 2.0]])
    
    # Forward pass
    output = layer.forward(x)
    
    # Assume loss is sum of outputs (simple case)
    loss = np.sum(output)
    print(f"Forward output: {output}")
    print(f"Loss (sum of output): {loss}")
    print()
    
    # Backward pass with gradient = 1 for all outputs
    grad_output = np.ones_like(output)
    grad_input = layer.backward(grad_output)
    
    print(f"Analytical gradient (weights):\n{layer.grad_weights}")
    print()
    
    # Numerical gradient check for weights
    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.weights)
    
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            # Perturb weight
            layer.weights[i, j] += epsilon
            loss_plus = np.sum(layer.forward(x))
            
            layer.weights[i, j] -= 2 * epsilon
            loss_minus = np.sum(layer.forward(x))
            
            # Restore weight
            layer.weights[i, j] += epsilon
            
            # Compute numerical gradient
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    print(f"Numerical gradient (weights):\n{numerical_grad}")
    print()
    
    # Check if gradients match
    diff = np.abs(layer.grad_weights - numerical_grad)
    max_diff = np.max(diff)
    print(f"Maximum difference: {max_diff}")
    
    if max_diff < 1e-5:
        print("✓ Gradients match! Backward pass is correct.")
    else:
        print("✗ Gradients don't match. There might be an error.")
    print()


def test_parameter_update():
    """Test parameter updates (simulating one optimization step)."""
    print("Test 5: Parameter Update")
    print("-" * 60)

    layer = Linear(input_dim=2, output_dim=2)

    # Save original weights
    original_weights = layer.weights.copy()
    original_bias = layer.bias.copy()

    print("Original weights:")
    print(original_weights)
    print()

    # Forward and backward pass
    x = np.random.randn(3, 2)
    output = layer.forward(x)
    grad_output = np.random.randn(3, 2)
    grad_input = layer.backward(grad_output)

    print("Gradients computed:")
    print(f"Grad weights:\n{layer.grad_weights}")
    print(f"Grad bias: {layer.grad_bias}")
    print()

    # Simulate parameter update (gradient descent)
    learning_rate = 0.01
    layer.weights -= learning_rate * layer.grad_weights
    layer.bias -= learning_rate * layer.grad_bias

    print("Updated weights:")
    print(layer.weights)
    print()

    print("Weight change:")
    print(layer.weights - original_weights)
    print()

    # Get parameters
    params = layer.get_parameters()
    print(f"Number of parameter groups: {len(params)}")
    for i, (param, grad) in enumerate(params):
        print(f"  Parameter {i+1} shape: {param.shape}")
    print()


def test_positional_encoding_basic():
    """Test basic positional encoding."""
    print("Test 1: Basic Positional Encoding")
    print("-" * 60)

    max_seq_len = 10
    embed_dim = 8
    pos_enc = PositionalEncoding(max_seq_len, embed_dim)

    print(f"Positional Encoding: {pos_enc}")
    print(f"Encoding matrix shape: {pos_enc.encodings.shape}")
    print()

    # Create embedded tokens
    batch_size = 2
    seq_len = 5
    embedded = np.random.randn(batch_size, seq_len, embed_dim)

    print(f"Input embedded tokens shape: {embedded.shape}")

    # Forward pass
    output = pos_enc.forward(embedded)

    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {output.shape == embedded.shape}")
    print()

    # Check that different positions get different encodings
    print("Positional encodings for first 3 positions:")
    for i in range(3):
        print(f"  Position {i}: {pos_enc.encodings[i, :4]} ...")
    print()


def test_positional_encoding_properties():
    """Test mathematical properties of positional encoding."""
    print("Test 2: Positional Encoding Properties")
    print("-" * 60)

    max_seq_len = 50
    embed_dim = 64
    pos_enc = PositionalEncoding(max_seq_len, embed_dim)

    # Check that encodings are bounded
    max_val = np.max(np.abs(pos_enc.encodings))
    print(f"Max absolute value in encodings: {max_val:.4f}")
    print(f"Values bounded by 1.0: {max_val <= 1.0}")
    print()

    # Check uniqueness: each position should have unique encoding
    unique_positions = set()
    for i in range(min(10, max_seq_len)):
        encoding_tuple = tuple(pos_enc.encodings[i])
        unique_positions.add(encoding_tuple)

    print(f"Checked {min(10, max_seq_len)} positions")
    print(f"All positions have unique encodings: {len(unique_positions) == min(10, max_seq_len)}")
    print()


def test_positional_encoding_gradient():
    """Test gradient flow through positional encoding."""
    print("Test 3: Positional Encoding Gradient Flow")
    print("-" * 60)

    pos_enc = PositionalEncoding(max_seq_len=10, embed_dim=8)

    # Forward pass
    embedded = np.random.randn(2, 5, 8)
    output = pos_enc.forward(embedded)

    # Backward pass
    grad_output = np.ones_like(output)
    grad_input = pos_enc.backward(grad_output)

    print(f"Gradient flows through unchanged: {np.allclose(grad_input, grad_output)}")
    print(f"No learnable parameters: {len(pos_enc.get_parameters()) == 0}")
    print()


def test_layer_norm_basic():
    """Test basic layer normalization."""
    print("Test 1: Basic Layer Normalization")
    print("-" * 60)

    normalized_shape = 8
    layer_norm = LayerNorm(normalized_shape)

    print(f"Layer Norm: {layer_norm}")
    print(f"Gamma (scale) shape: {layer_norm.gamma.shape}")
    print(f"Beta (shift) shape: {layer_norm.beta.shape}")
    print()

    # Create input with non-zero mean and variance
    x = np.random.randn(2, 5, 8) * 3.0 + 10.0  # mean≈10, std≈3

    print(f"Input shape: {x.shape}")
    print(f"Input mean (along last dim): {np.mean(x, axis=-1)}")
    print(f"Input std (along last dim): {np.std(x, axis=-1)}")
    print()

    # Forward pass
    output = layer_norm.forward(x)

    print(f"Output shape: {output.shape}")
    print(f"Output mean (should be ≈0): {np.mean(output, axis=-1)}")
    print(f"Output std (should be ≈1): {np.std(output, axis=-1)}")
    print()


def test_layer_norm_gradient():
    """Test layer norm gradients."""
    print("Test 2: Layer Normalization Gradients")
    print("-" * 60)

    layer_norm = LayerNorm(normalized_shape=4)

    # Forward pass
    x = np.random.randn(2, 3, 4)
    output = layer_norm.forward(x)

    # Backward pass
    grad_output = np.random.randn(2, 3, 4)
    grad_input = layer_norm.backward(grad_output)

    print(f"Input shape: {x.shape}")
    print(f"Grad input shape: {grad_input.shape}")
    print(f"Grad gamma shape: {layer_norm.grad_gamma.shape}")
    print(f"Grad beta shape: {layer_norm.grad_beta.shape}")
    print()

    # Get parameters
    params = layer_norm.get_parameters()
    print(f"Number of parameter groups: {len(params)}")
    print(f"Gamma and beta are trainable: {len(params) == 2}")
    print()


def test_relu_basic():
    """Test basic ReLU activation."""
    print("Test 1: Basic ReLU")
    print("-" * 60)

    relu = ReLU()
    print(f"ReLU: {relu}")
    print()

    # Test with mixed positive/negative values
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                  [3.0, -0.5, 1.5, -3.0, 0.5]])

    print(f"Input:\n{x}")

    # Forward pass
    output = relu.forward(x)

    print(f"\nOutput (negative values zeroed):\n{output}")
    print()

    # Check that negative values are zero
    negative_zeroed = np.all(output[x < 0] == 0)
    positive_unchanged = np.all(output[x > 0] == x[x > 0])

    print(f"Negative values zeroed: {negative_zeroed}")
    print(f"Positive values unchanged: {positive_unchanged}")
    print()


def test_relu_gradient():
    """Test ReLU gradient."""
    print("Test 2: ReLU Gradient")
    print("-" * 60)

    relu = ReLU()

    # Forward pass
    x = np.array([[-1.0, 2.0, -3.0, 4.0]])
    output = relu.forward(x)

    # Backward pass
    grad_output = np.ones_like(output)
    grad_input = relu.backward(grad_output)

    print(f"Input: {x}")
    print(f"Gradient output: {grad_output}")
    print(f"Gradient input: {grad_input}")
    print()

    # Gradient should be 1 where x > 0, else 0
    expected_grad = (x > 0).astype(float)
    print(f"Expected gradient: {expected_grad}")
    print(f"Gradients match: {np.allclose(grad_input, expected_grad)}")
    print()


def test_softmax_basic():
    """Test basic softmax."""
    print("Test 1: Basic Softmax")
    print("-" * 60)

    softmax = Softmax()
    print(f"Softmax: {softmax}")
    print()

    # Test with simple logits
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [0.1, 0.2, 0.3, 0.4]])

    print(f"Input (logits):\n{x}")

    # Forward pass
    output = softmax.forward(x)

    print(f"\nOutput (probabilities):\n{output}")
    print()

    # Check properties
    sums = np.sum(output, axis=-1)
    all_positive = np.all(output >= 0)
    sums_to_one = np.allclose(sums, 1.0)

    print(f"All values positive: {all_positive}")
    print(f"Sums to 1.0 along axis: {sums_to_one}")
    print(f"Sums: {sums}")
    print()


def test_softmax_numerical_stability():
    """Test softmax numerical stability."""
    print("Test 2: Softmax Numerical Stability")
    print("-" * 60)

    softmax = Softmax()

    # Test with large values (would overflow without stability trick)
    x_large = np.array([[1000.0, 1001.0, 1002.0]])

    print(f"Input (large values):\n{x_large}")

    # Forward pass (should not produce NaN or Inf)
    output = softmax.forward(x_large)

    print(f"Output:\n{output}")
    print(f"No NaN: {not np.any(np.isnan(output))}")
    print(f"No Inf: {not np.any(np.isinf(output))}")
    print(f"Sums to 1.0: {np.allclose(np.sum(output, axis=-1), 1.0)}")
    print()


def test_softmax_gradient():
    """Test softmax gradient."""
    print("Test 3: Softmax Gradient")
    print("-" * 60)

    softmax = Softmax()

    # Forward pass
    x = np.array([[1.0, 2.0, 3.0]])
    output = softmax.forward(x)

    print(f"Input: {x}")
    print(f"Output (probs): {output}")
    print()

    # Backward pass
    grad_output = np.array([[1.0, 0.0, 0.0]])  # Gradient for first class
    grad_input = softmax.backward(grad_output)

    print(f"Gradient output: {grad_output}")
    print(f"Gradient input: {grad_input}")
    print()

    # Numerical gradient check
    epsilon = 1e-5
    numerical_grad = np.zeros_like(x)

    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[0, i] += epsilon
        out_plus = softmax.forward(x_plus)
        loss_plus = np.sum(out_plus * grad_output)

        x_minus = x.copy()
        x_minus[0, i] -= epsilon
        out_minus = softmax.forward(x_minus)
        loss_minus = np.sum(out_minus * grad_output)

        numerical_grad[0, i] = (loss_plus - loss_minus) / (2 * epsilon)

    print(f"Numerical gradient: {numerical_grad}")
    print(f"Gradients match: {np.allclose(grad_input, numerical_grad, atol=1e-5)}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LAYERS MODULE TEST - PHASE 2 COMPLETE")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Test Embedding Layer
    print("\n" + "=" * 60)
    print("EMBEDDING LAYER TESTS")
    print("=" * 60)
    print()

    test_embedding_basic()
    test_embedding_gradients()
    test_embedding_same_token()
    test_embedding_with_padding()

    # Test Linear Layer
    print("\n" + "=" * 60)
    print("LINEAR LAYER TESTS")
    print("=" * 60)
    print()

    test_linear_2d()
    test_linear_3d()
    test_linear_no_bias()
    test_gradient_numerical()
    test_parameter_update()

    # Test Positional Encoding
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING TESTS")
    print("=" * 60)
    print()

    test_positional_encoding_basic()
    test_positional_encoding_properties()
    test_positional_encoding_gradient()

    # Test Layer Normalization
    print("\n" + "=" * 60)
    print("LAYER NORMALIZATION TESTS")
    print("=" * 60)
    print()

    test_layer_norm_basic()
    test_layer_norm_gradient()

    # Test ReLU
    print("\n" + "=" * 60)
    print("RELU ACTIVATION TESTS")
    print("=" * 60)
    print()

    test_relu_basic()
    test_relu_gradient()

    # Test Softmax
    print("\n" + "=" * 60)
    print("SOFTMAX ACTIVATION TESTS")
    print("=" * 60)
    print()

    test_softmax_basic()
    test_softmax_numerical_stability()
    test_softmax_gradient()

    # Summary
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Phase 2 - Core Components COMPLETE:")
    print("  ✓ Embedding layer: token lookup and gradient accumulation")
    print("  ✓ Linear layer: forward/backward passes with numerical verification")
    print("  ✓ Positional Encoding: sinusoidal encodings for position information")
    print("  ✓ Layer Normalization: mean/variance normalization with learnable params")
    print("  ✓ ReLU activation: non-linearity with correct gradients")
    print("  ✓ Softmax activation: probability distribution with numerical stability")
    print()
    print("All core components ready for building the transformer!")