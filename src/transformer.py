"""
Transformer Architecture Module

This module implements the complete transformer architecture for our digit operations task.

The Transformer Revolution:
    The "Attention Is All You Need" paper (Vaswani et al., 2017) introduced the transformer
    architecture that has become the foundation for modern LLMs (GPT, BERT, etc.).

    Key innovations:
    1. Self-attention replaces recurrence (enables parallelization)
    2. Positional encoding handles sequential information
    3. Residual connections enable training deep networks
    4. Layer normalization stabilizes training
    5. Feed-forward networks add non-linear transformations

Architecture Flow:
    Input Tokens
        ↓
    Embedding + Positional Encoding
        ↓
    Transformer Block 1
        ├─ Multi-Head Attention
        ├─ Add & Norm (residual)
        ├─ Feed-Forward Network
        └─ Add & Norm (residual)
        ↓
    Transformer Block 2
        └─ (same structure)
        ↓
    Output Projection
        ↓
    Softmax → Predictions

This module implements:
    1. FeedForward: Position-wise feed-forward network
    2. TransformerBlock: Complete transformer block with attention + FFN
    3. Transformer: Full encoder-style transformer model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layers'))

import numpy as np
from layers import (
    Embedding, Linear, PositionalEncoding, LayerNorm,
    ReLU, Softmax, MultiHeadAttention
)


class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN).

    This is a simple two-layer neural network applied independently to each position.
    "Position-wise" means the same network is applied to each token separately.

    Why Feed-Forward Networks in Transformers?
        Attention is great at mixing information between positions, but it's entirely
        linear operations (weighted sums). FFN adds crucial non-linearity that enables
        the model to learn complex transformations.

        Think of it as:
        - Attention: "What information should I gather?"
        - FFN: "How should I transform this information?"

    Architecture:
        input → Linear(embed_dim → ffn_dim) → ReLU → Linear(ffn_dim → embed_dim) → output

    Why Two Layers?
        - First layer: Project to higher dimension (expansion)
        - Second layer: Project back to original dimension (compression)
        - This "expand-compress" pattern lets the network learn rich representations

    Why ffn_dim > embed_dim?
        Common practice: ffn_dim = 4 × embed_dim
        - Higher dimension provides more representational capacity
        - The bottleneck (compressing back down) forces learning of useful features
        - Similar to autoencoder architecture

    Why ReLU Activation?
        - Introduces non-linearity (crucial for learning complex functions)
        - Computationally efficient
        - Doesn't saturate for positive values (healthy gradients)
        - Original transformer used ReLU; modern variants use GELU or SiLU

    Position-wise vs Global:
        Each token is processed independently (no mixing between positions here).
        This is by design:
        - Attention handles position interactions
        - FFN handles per-position transformations
        - Separation of concerns leads to cleaner architecture

    Args:
        embed_dim (int): Input/output dimension (e.g., 64)
        ffn_dim (int): Hidden dimension (e.g., 256 = 4 × 64)
        dropout (float): Dropout probability (not implemented for simplicity)

    Attributes:
        linear1 (Linear): First linear layer (expand)
        relu (ReLU): Non-linear activation
        linear2 (Linear): Second linear layer (compress)

    Example:
        >>> ffn = FeedForward(embed_dim=64, ffn_dim=256)
        >>> x = np.random.randn(2, 10, 64)  # (batch, seq_len, embed_dim)
        >>> output = ffn.forward(x)
        >>> output.shape  # (2, 10, 64) - same as input
    """

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Initialize Feed-Forward Network.

        Why This Structure?
            Two linear layers with ReLU activation is the standard FFN design.
            It's simple but remarkably effective.

        Alternative Architectures (not implemented):
            - Gated Linear Units (GLU): x * sigmoid(Wx)
            - Swish/SiLU activation: x * sigmoid(x)
            - More than 2 layers (uncommon, diminishing returns)

        Args:
            embed_dim (int): Embedding dimension
            ffn_dim (int): Feed-forward hidden dimension (typically 4× embed_dim)
            dropout (float): Dropout rate (we don't implement dropout for simplicity)
        """
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        # First linear layer: embed_dim → ffn_dim (expansion)
        self.linear1 = Linear(input_dim=embed_dim, output_dim=ffn_dim, use_bias=True)

        # Non-linear activation
        self.relu = ReLU()

        # Second linear layer: ffn_dim → embed_dim (compression)
        self.linear2 = Linear(input_dim=ffn_dim, output_dim=embed_dim, use_bias=True)

        # Cache for backward pass
        self.linear1_output_cache = None
        self.relu_output_cache = None

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Computation:
            1. Linear1: (batch, seq_len, embed_dim) → (batch, seq_len, ffn_dim)
            2. ReLU: Apply non-linearity element-wise
            3. Linear2: (batch, seq_len, ffn_dim) → (batch, seq_len, embed_dim)

        Why This Order?
            Expand → Activate → Compress is a standard pattern in deep learning.
            The expansion gives the network more "room" to learn complex patterns,
            then compression forces it to keep only the most useful features.

        Mathematical View:
            output = W2 · ReLU(W1 · x + b1) + b2
            Where W1 is (embed_dim, ffn_dim) and W2 is (ffn_dim, embed_dim)

        Args:
            x (np.ndarray): Input tensor, shape (batch, seq_len, embed_dim)

        Returns:
            np.ndarray: Output tensor, shape (batch, seq_len, embed_dim)
        """
        # First linear transformation (expansion)
        linear1_output = self.linear1.forward(x)
        self.linear1_output_cache = linear1_output

        # Non-linear activation
        relu_output = self.relu.forward(linear1_output)
        self.relu_output_cache = relu_output

        # Second linear transformation (compression)
        output = self.linear2.forward(relu_output)

        return output

    def backward(self, grad_output):
        """
        Backward pass through feed-forward network.

        Gradient Flow:
            grad_output → [linear2 backward] → [relu backward] → [linear1 backward] → grad_input

        Chain Rule Application:
            We apply the chain rule through each layer in reverse order.
            Each layer computes its parameter gradients and passes gradients to previous layer.

        Why Reverse Order?
            Backpropagation naturally flows from output to input.
            Each layer needs gradients from the next layer to compute its own gradients.

        Args:
            grad_output (np.ndarray): Gradient from next layer, shape (batch, seq_len, embed_dim)

        Returns:
            np.ndarray: Gradient with respect to input, shape (batch, seq_len, embed_dim)
        """
        # Backward through second linear layer
        grad_relu = self.linear2.backward(grad_output)

        # Backward through ReLU activation
        grad_linear1 = self.relu.backward(grad_relu)

        # Backward through first linear layer
        grad_input = self.linear1.backward(grad_linear1)

        return grad_input

    def get_parameters(self):
        """
        Get all trainable parameters and their gradients.

        Returns:
            list: Concatenated parameters from both linear layers
                  Format: [(param, grad), (param, grad), ...]
        """
        # Collect parameters from both linear layers
        params = []
        params.extend(self.linear1.get_parameters())
        params.extend(self.linear2.get_parameters())
        return params

    def zero_grad(self):
        """Reset all gradients to None."""
        self.linear1.zero_grad()
        self.linear2.zero_grad()
        self.relu.zero_grad()

    def __repr__(self):
        """String representation for debugging."""
        return f"FeedForward(embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim})"


class TransformerBlock:
    """
    Complete Transformer Block with self-attention and feed-forward network.

    This is the fundamental building block of transformers. A transformer is simply
    a stack of these blocks (typically 6-12 for BERT, up to 96 for GPT-3).

    Architecture (Add & Norm pattern):
        ```
        Input (x)
            │
            ├─────────────────┐
            │                 │
            ↓                 │
        Multi-Head        (residual)
        Attention             │
            │                 │
            ↓                 │
        Add ←─────────────────┘
            │
            ↓
        Layer Norm
            │
            ├─────────────────┐
            │                 │
            ↓                 │
        Feed-Forward      (residual)
        Network               │
            │                 │
            ↓                 │
        Add ←─────────────────┘
            │
            ↓
        Layer Norm
            │
            ↓
        Output
        ```

    Why Residual Connections (Skip Connections)?
        Problem: Deep networks suffer from vanishing gradients.
        Solution: Add input directly to output (x + F(x) instead of just F(x))

        Benefits:
        1. Gradient Flow: Gradients can flow directly through the residual path
        2. Identity Initialization: Network can learn to be identity (F(x)=0) initially
        3. Easier Optimization: Network learns "refinements" rather than full transformation
        4. Enables Depth: Allows training 100+ layer networks

        Mathematical Insight:
            Instead of learning H(x), learn F(x) = H(x) - x (residual)
            Then output is x + F(x) = H(x)
            Learning the residual is often easier than learning the full mapping

    Why Layer Normalization?
        - Stabilizes training by normalizing activations
        - Reduces internal covariate shift
        - Applied AFTER residual addition (Post-LN)
        - Alternative: Pre-LN (normalize before attention) - more stable for deep models

    Post-LN vs Pre-LN:
        Post-LN (what we implement): x = LayerNorm(x + Attention(x))
        - Original transformer design
        - Works well for shallow models (2-6 layers)

        Pre-LN: x = x + Attention(LayerNorm(x))
        - More stable for very deep models
        - Used in GPT-2, GPT-3
        - We use Post-LN for simplicity

    Why This Two-Sub-Layer Design?
        1. Attention: Mix information between positions
        2. FFN: Transform information at each position
        This separation of concerns makes the model interpretable and effective

    Args:
        embed_dim (int): Embedding dimension (e.g., 64)
        num_heads (int): Number of attention heads (e.g., 4)
        ffn_dim (int): Feed-forward hidden dimension (e.g., 256)
        dropout (float): Dropout rate (not implemented)

    Attributes:
        attention (MultiHeadAttention): Multi-head self-attention
        norm1 (LayerNorm): Layer norm after attention
        ffn (FeedForward): Feed-forward network
        norm2 (LayerNorm): Layer norm after FFN

    Example:
        >>> block = TransformerBlock(embed_dim=64, num_heads=4, ffn_dim=256)
        >>> x = np.random.randn(2, 10, 64)  # (batch, seq_len, embed_dim)
        >>> output = block.forward(x)
        >>> output.shape  # (2, 10, 64) - same as input
    """

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        """
        Initialize Transformer Block.

        Why These Components?
            - Attention: Handles position interactions
            - FFN: Handles per-position transformations
            - Layer Norm: Stabilizes training
            - Residual: Enables gradient flow

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ffn_dim (int): Feed-forward hidden dimension
            dropout (float): Dropout rate (not implemented for simplicity)
        """
        self.embed_dim = embed_dim

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # First layer normalization (after attention)
        self.norm1 = LayerNorm(normalized_shape=embed_dim)

        # Position-wise feed-forward network
        self.ffn = FeedForward(embed_dim=embed_dim, ffn_dim=ffn_dim)

        # Second layer normalization (after FFN)
        self.norm2 = LayerNorm(normalized_shape=embed_dim)

        # Cache for backward pass
        self.input_cache = None
        self.attention_output_cache = None
        self.residual1_cache = None
        self.norm1_output_cache = None
        self.ffn_output_cache = None
        self.residual2_cache = None

    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.

        Complete Flow:
            1. x_attn = MultiHeadAttention(x)
            2. x = LayerNorm(x + x_attn)       # First Add & Norm
            3. x_ffn = FeedForward(x)
            4. x = LayerNorm(x + x_ffn)        # Second Add & Norm

        Why This Order?
            This is the Post-LN architecture from the original transformer paper.
            The residual connection preserves the original signal, while the
            sub-layer (attention or FFN) learns refinements.

        Residual Connection Magic:
            Without residual: x = F(x) - network must learn entire transformation
            With residual: x = x + F(x) - network only learns the "delta"
            This makes optimization much easier!

        Args:
            x (np.ndarray): Input tensor, shape (batch, seq_len, embed_dim)
            mask (np.ndarray, optional): Attention mask

        Returns:
            np.ndarray: Output tensor, shape (batch, seq_len, embed_dim)
        """
        # Cache input for backward pass
        self.input_cache = x

        # Sub-layer 1: Multi-head self-attention
        attention_output, _ = self.attention.forward(x, mask=mask)
        self.attention_output_cache = attention_output

        # Residual connection 1
        residual1 = x + attention_output
        self.residual1_cache = residual1

        # Layer normalization 1
        norm1_output = self.norm1.forward(residual1)
        self.norm1_output_cache = norm1_output

        # Sub-layer 2: Feed-forward network
        ffn_output = self.ffn.forward(norm1_output)
        self.ffn_output_cache = ffn_output

        # Residual connection 2
        residual2 = norm1_output + ffn_output
        self.residual2_cache = residual2

        # Layer normalization 2
        output = self.norm2.forward(residual2)

        return output

    def backward(self, grad_output):
        """
        Backward pass through transformer block.

        This is complex because we need to handle:
        1. Gradients through layer normalization
        2. Gradients through residual connections (split into two paths)
        3. Gradients through FFN and attention

        Residual Backward Pass:
            For y = x + F(x), we have:
            ∂L/∂x = ∂L/∂y + ∂L/∂F(x)

            The gradient splits into two paths:
            - Direct path: gradient flows directly through the addition
            - Indirect path: gradient flows through the sub-layer F

        Why Residual Helps Gradients:
            The direct path (∂L/∂y) ensures gradients always have a "highway"
            to flow backward, even if F's gradients vanish.

        Args:
            grad_output (np.ndarray): Gradient from next layer

        Returns:
            np.ndarray: Gradient with respect to input
        """
        # Backward through second layer norm
        grad_residual2 = self.norm2.backward(grad_output)

        # Backward through second residual connection (splits gradient)
        grad_ffn_output = grad_residual2  # Gradient to FFN
        grad_norm1_output_from_residual2 = grad_residual2  # Gradient bypassing FFN

        # Backward through feed-forward network
        grad_norm1_output_from_ffn = self.ffn.backward(grad_ffn_output)

        # Combine gradients at norm1 output
        grad_norm1_output = grad_norm1_output_from_residual2 + grad_norm1_output_from_ffn

        # Backward through first layer norm
        grad_residual1 = self.norm1.backward(grad_norm1_output)

        # Backward through first residual connection (splits gradient)
        grad_attention_output = grad_residual1  # Gradient to attention
        grad_input_from_residual1 = grad_residual1  # Gradient bypassing attention

        # Backward through multi-head attention
        grad_input_from_attention = self.attention.backward(grad_attention_output)

        # Combine gradients at input
        grad_input = grad_input_from_residual1 + grad_input_from_attention

        return grad_input

    def get_parameters(self):
        """
        Get all trainable parameters from all sub-layers.

        Returns:
            list: Parameters from attention, FFN, and layer norms
        """
        params = []
        params.extend(self.attention.get_parameters())
        params.extend(self.norm1.get_parameters())
        params.extend(self.ffn.get_parameters())
        params.extend(self.norm2.get_parameters())
        return params

    def zero_grad(self):
        """Reset all gradients in all sub-layers."""
        self.attention.zero_grad()
        self.norm1.zero_grad()
        self.ffn.zero_grad()
        self.norm2.zero_grad()

    def __repr__(self):
        """String representation for debugging."""
        return (f"TransformerBlock(embed_dim={self.embed_dim}, "
                f"num_heads={self.attention.num_heads}, "
                f"ffn_dim={self.ffn.ffn_dim})")


class Transformer:
    """
    Complete Transformer Model for sequence classification.

    This is an encoder-only transformer designed for our digit operations task.
    It takes a sequence of tokens and outputs a single prediction.

    Architecture:
        ```
        Input Token IDs (batch, seq_len)
            ↓
        Embedding Layer → Dense vectors (batch, seq_len, embed_dim)
            ↓
        Positional Encoding → Add position info
            ↓
        Transformer Block 1
            ├─ Self-Attention
            └─ Feed-Forward
            ↓
        Transformer Block 2
            ├─ Self-Attention
            └─ Feed-Forward
            ↓
        ... (more blocks)
            ↓
        Pool → Take first token (batch, embed_dim)
            ↓
        Output Projection → Logits (batch, vocab_size)
        ```

    Encoder vs Decoder:
        - Encoder (what we implement): Bidirectional attention, sees full sequence
        - Decoder: Causal attention, only sees previous tokens
        - Our task doesn't require generation, so encoder-only is perfect

    Why First Token for Classification?
        We pool the representation of the first token to get sequence representation.
        Alternatives:
        - Mean pooling: Average all tokens
        - Max pooling: Take max across tokens
        - First token (CLS): Standard in BERT
        - Last token: Common in GPT-style models

        For our task, mean pooling might be better (considers all tokens equally),
        but first token is simpler and works well.

    Args:
        vocab_size (int): Size of vocabulary (e.g., 20)
        embed_dim (int): Embedding dimension (e.g., 64)
        num_heads (int): Number of attention heads (e.g., 4)
        num_layers (int): Number of transformer blocks (e.g., 2)
        ffn_dim (int): Feed-forward hidden dimension (e.g., 256)
        max_seq_len (int): Maximum sequence length (e.g., 50)
        dropout (float): Dropout rate (not implemented)

    Example:
        >>> model = Transformer(vocab_size=20, embed_dim=64, num_heads=4, num_layers=2)
        >>> tokens = np.array([[15, 17, 7, 19, 5]])  # (batch=1, seq_len=5)
        >>> logits = model.forward(tokens)
        >>> logits.shape  # (1, 20) - probabilities over vocabulary
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim, max_seq_len=50, dropout=0.1):
        """
        Initialize complete transformer model.

        Why These Hyperparameters?
            - vocab_size: Determined by task (20 tokens for our problem)
            - embed_dim: Trade-off between capacity and speed (64 is small but sufficient)
            - num_heads: More heads = more relationship types (4 is standard for small models)
            - num_layers: Depth enables complex representations (2 is sufficient for toy problem)
            - ffn_dim: Typically 4× embed_dim for representational capacity
            - max_seq_len: Must accommodate longest input (50 is plenty for our task)

        Args:
            vocab_size (int): Vocabulary size
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads per layer
            num_layers (int): Number of transformer blocks
            ffn_dim (int): Feed-forward hidden dimension
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout probability (not implemented)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Token embedding layer
        self.embedding = Embedding(vocab_size=vocab_size, embed_dim=embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, embed_dim=embed_dim)

        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ]

        # Output projection layer (embed_dim → vocab_size)
        self.output_projection = Linear(input_dim=embed_dim, output_dim=vocab_size, use_bias=True)

        # Softmax for converting logits to probabilities
        self.softmax = Softmax(axis=-1)

        # Cache for backward pass
        self.input_cache = None
        self.embedded_cache = None
        self.pos_encoded_cache = None
        self.block_outputs_cache = []
        self.pooled_cache = None

    def forward(self, x, mask=None, return_logits=False):
        """
        Forward pass through complete transformer.

        Complete Flow:
            1. Token IDs → Embeddings
            2. Add positional encodings
            3. Pass through N transformer blocks
            4. Pool to single vector (take first token)
            5. Project to vocabulary size
            6. Apply softmax for probabilities

        Why This Sequence?
            Each step builds on the previous:
            - Embeddings convert discrete to continuous
            - Position encoding adds sequence order info
            - Transformer blocks learn relationships
            - Pooling aggregates to fixed size
            - Projection maps to output space

        Args:
            x (np.ndarray): Token IDs, shape (batch, seq_len)
            mask (np.ndarray, optional): Attention mask
            return_logits (bool): If True, return logits instead of probabilities

        Returns:
            np.ndarray: Predictions, shape (batch, vocab_size)
        """
        # Cache input
        self.input_cache = x

        # Step 1: Embedding lookup
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding.forward(x)
        self.embedded_cache = embedded

        # Step 2: Add positional encoding
        # (batch, seq_len, embed_dim) → (batch, seq_len, embed_dim)
        pos_encoded = self.pos_encoding.forward(embedded)
        self.pos_encoded_cache = pos_encoded

        # Step 3: Pass through transformer blocks
        hidden = pos_encoded
        self.block_outputs_cache = []

        for block in self.blocks:
            hidden = block.forward(hidden, mask=mask)
            self.block_outputs_cache.append(hidden)

        # Step 4: Pooling - take first token representation
        # (batch, seq_len, embed_dim) → (batch, embed_dim)
        # Alternative: could use mean pooling or last token
        pooled = hidden[:, 0, :]  # Take first token
        self.pooled_cache = pooled

        # Step 5: Output projection
        # (batch, embed_dim) → (batch, vocab_size)
        logits = self.output_projection.forward(pooled)

        if return_logits:
            return logits

        # Step 6: Apply softmax to get probabilities
        # (batch, vocab_size) → (batch, vocab_size)
        probs = self.softmax.forward(logits)

        return probs

    def backward(self, grad_output):
        """
        Backward pass through complete transformer.

        This implements backpropagation through all layers in reverse order.

        Gradient Flow:
            grad_output → softmax → output_proj → unpooling → blocks[N-1] → ... →
            blocks[0] → pos_encoding → embedding

        Why Reverse Order?
            Backpropagation naturally flows from outputs to inputs, computing
            gradients layer by layer using the chain rule.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t. output

        Returns:
            np.ndarray: Gradient w.r.t. input (typically not used for discrete tokens)
        """
        # Backward through softmax
        grad_logits = self.softmax.backward(grad_output)

        # Backward through output projection
        grad_pooled = self.output_projection.backward(grad_logits)

        # Backward through pooling (unpooling)
        # We took first token, so gradient goes only to first position
        batch_size, seq_len, embed_dim = self.block_outputs_cache[-1].shape
        grad_hidden = np.zeros((batch_size, seq_len, embed_dim))
        grad_hidden[:, 0, :] = grad_pooled

        # Backward through transformer blocks (in reverse order)
        for block in reversed(self.blocks):
            grad_hidden = block.backward(grad_hidden)

        # Backward through positional encoding (no parameters, just pass through)
        grad_embedded = self.pos_encoding.backward(grad_hidden)

        # Backward through embedding
        grad_input = self.embedding.backward(grad_embedded)

        return grad_input

    def get_parameters(self):
        """
        Get all trainable parameters from all components.

        Returns:
            list: All parameters and their gradients
        """
        params = []
        params.extend(self.embedding.get_parameters())
        # Positional encoding has no learnable parameters
        for block in self.blocks:
            params.extend(block.get_parameters())
        params.extend(self.output_projection.get_parameters())
        return params

    def zero_grad(self):
        """Reset all gradients in all components."""
        self.embedding.zero_grad()
        self.pos_encoding.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.output_projection.zero_grad()
        self.softmax.zero_grad()

    def __repr__(self):
        """String representation for debugging."""
        return (f"Transformer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"num_layers={self.num_layers})")
