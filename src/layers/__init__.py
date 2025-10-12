"""
Neural Network Layers Package

This package provides fundamental neural network layer implementations from scratch
using only NumPy. All layers support both forward and backward passes for training.

Modules:
    embedding: Token embedding layer
    linear: Fully connected (dense) layer
    positional_encoding: Sinusoidal positional encoding for transformers
    normalization: Layer normalization
    activations: ReLU and Softmax activation functions
    attention: Scaled dot-product attention and multi-head attention

Usage:
    from layers import (Embedding, Linear, PositionalEncoding, LayerNorm,
                        ReLU, Softmax, MultiHeadAttention)

    # Create layers
    embedding = Embedding(vocab_size=20, embed_dim=64)
    pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
    linear = Linear(input_dim=64, output_dim=128)
    layer_norm = LayerNorm(normalized_shape=128)
    relu = ReLU()
    softmax = Softmax()
    attention = MultiHeadAttention(embed_dim=64, num_heads=4)
"""

from .embedding import Embedding
from .linear import Linear
from .positional_encoding import PositionalEncoding
from .normalization import LayerNorm
from .activations import ReLU, Softmax
from .attention import MultiHeadAttention, scaled_dot_product_attention

__all__ = [
    'Embedding',
    'Linear',
    'PositionalEncoding',
    'LayerNorm',
    'ReLU',
    'Softmax',
    'MultiHeadAttention',
    'scaled_dot_product_attention',
]

__version__ = '0.3.0'
