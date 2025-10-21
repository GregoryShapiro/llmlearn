"""
Decoder-Only Transformer for Autoregressive Generation

This is like GPT - it generates sequences one token at a time.

Key differences from encoder (transformer.py):
1. Uses causal masking (can't see future)
2. Outputs logits for ALL positions (not just first)
3. Designed for next-token prediction
"""

import numpy as np
from layers.embedding import Embedding
from layers.positional_encoding import PositionalEncoding
from layers.attention import MultiHeadAttention
from layers.linear import Linear
from layers.normalization import LayerNorm
from layers.activations import ReLU
from decoder_utils import create_causal_mask


class TransformerDecoderBlock:
    """
    Single decoder block with causal self-attention.

    Same as encoder block but with causal masking.
    """

    def __init__(self, embed_dim, num_heads, ffn_dim):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)

        self.ffn = [
            Linear(embed_dim, ffn_dim),
            ReLU(),
            Linear(ffn_dim, embed_dim)
        ]
        self.norm2 = LayerNorm(embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x, causal_mask=None):
        """Forward with causal masking."""
        # Self-attention with causal mask
        attn_out, _ = self.attention.forward(x, mask=causal_mask)
        x = self.norm1.forward(x + attn_out)

        # FFN
        ffn_out = x
        for layer in self.ffn:
            ffn_out = layer.forward(ffn_out)
        x = self.norm2.forward(x + ffn_out)

        return x

    def backward(self, grad):
        """Backward pass."""
        # Norm2 backward
        grad = self.norm2.backward(grad)

        # FFN backward
        grad_residual = grad
        for layer in reversed(self.ffn):
            grad = layer.backward(grad)
        grad = grad + grad_residual

        # Norm1 backward
        grad = self.norm1.backward(grad)

        # Attention backward
        grad_residual = grad
        grad = self.attention.backward(grad)
        grad = grad + grad_residual

        return grad

    def get_parameters(self):
        """Get all parameters."""
        params = []
        params.extend(self.attention.get_parameters())
        params.extend(self.norm1.get_parameters())
        for layer in self.ffn:
            if hasattr(layer, 'get_parameters'):
                params.extend(layer.get_parameters())
        params.extend(self.norm2.get_parameters())
        return params


class TransformerDecoder:
    """
    Decoder-only transformer for autoregressive generation.

    Architecture:
        Input Tokens → Embedding → Positional Encoding
                           ↓
                    Decoder Block 1 (causal attention)
                           ↓
                    Decoder Block 2 (causal attention)
                           ↓
                    Output Projection → Logits

    Output shape: (batch, seq_len, vocab_size)
        - Predicts next token for EVERY position
        - Use logits[:, i, :] to predict token at position i+1
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim, max_seq_len):
        """
        Initialize decoder transformer.

        Args:
            vocab_size (int): Size of vocabulary
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads per layer
            num_layers (int): Number of decoder blocks
            ffn_dim (int): Feed-forward network hidden dimension
            max_seq_len (int): Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)

        # Decoder blocks
        self.blocks = [
            TransformerDecoderBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]

        # Output projection (embed_dim -> vocab_size)
        self.output_projection = Linear(embed_dim, vocab_size)

        # Cache for backward
        self.input_cache = None
        self.embed_cache = None
        self.pos_cache = None
        self.block_outputs = None

    def forward(self, input_ids):
        """
        Forward pass.

        Args:
            input_ids (np.ndarray): Token IDs, shape (batch, seq_len)

        Returns:
            np.ndarray: Logits, shape (batch, seq_len, vocab_size)
                       logits[b, i, :] = distribution over next token at position i
        """
        self.input_cache = input_ids
        batch_size, seq_len = input_ids.shape

        # Create causal mask (same for all examples in batch)
        causal_mask = create_causal_mask(seq_len)  # (seq_len, seq_len)

        # Expand for batch: (seq_len, seq_len) -> (batch, seq_len, seq_len)
        # The attention code will add the head dimension automatically
        causal_mask = causal_mask[np.newaxis, :, :]
        causal_mask = np.repeat(causal_mask, batch_size, axis=0)

        # Embedding
        x = self.embedding.forward(input_ids)  # (batch, seq_len, embed_dim)
        self.embed_cache = x

        # Positional encoding
        x = self.pos_encoding.forward(x)  # (batch, seq_len, embed_dim)
        self.pos_cache = x

        # Decoder blocks
        self.block_outputs = []
        for block in self.blocks:
            x = block.forward(x, causal_mask=causal_mask)
            self.block_outputs.append(x)

        # Output projection to vocabulary
        logits = self.output_projection.forward(x)  # (batch, seq_len, vocab_size)

        return logits

    def backward(self, grad_logits):
        """
        Backward pass.

        Args:
            grad_logits (np.ndarray): Gradient of loss w.r.t logits
                                     Shape: (batch, seq_len, vocab_size)

        Returns:
            None (gradients stored in parameters)
        """
        # Output projection backward
        grad = self.output_projection.backward(grad_logits)

        # Decoder blocks backward
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        # Positional encoding backward
        grad = self.pos_encoding.backward(grad)

        # Embedding backward
        self.embedding.backward(grad)

    def get_parameters(self):
        """Get all trainable parameters and their gradients."""
        params = []
        params.extend(self.embedding.get_parameters())
        for block in self.blocks:
            params.extend(block.get_parameters())
        params.extend(self.output_projection.get_parameters())
        return params

    def __repr__(self):
        total_params = sum(p.size for p, _ in self.get_parameters())
        return (f"TransformerDecoder(\n"
                f"  vocab_size={self.vocab_size},\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  num_layers={len(self.blocks)},\n"
                f"  total_params={total_params:,}\n"
                f")")
