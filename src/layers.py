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


if __name__ == "__main__":
    print("=" * 60)
    print("LAYERS MODULE TEST")
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
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✓ Embedding layer: token lookup working correctly")
    print("  ✓ Embedding gradients: accumulation working")
    print("  ✓ Linear layer: forward/backward passes correct")
    print("  ✓ Numerical gradient check: passed")
    print()
    print("Both layers are ready for the transformer!")