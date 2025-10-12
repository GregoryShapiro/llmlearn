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

Usage:
    from layers import Embedding, Linear, PositionalEncoding, LayerNorm, ReLU, Softmax

    # Create layers
    embedding = Embedding(vocab_size=20, embed_dim=64)
    pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=64)
    linear = Linear(input_dim=64, output_dim=128)
    layer_norm = LayerNorm(normalized_shape=128)
    relu = ReLU()
    softmax = Softmax()
"""

from .embedding import Embedding
from .linear import Linear
from .positional_encoding import PositionalEncoding
from .normalization import LayerNorm
from .activations import ReLU, Softmax

__all__ = [
    'Embedding',
    'Linear',
    'PositionalEncoding',
    'LayerNorm',
    'ReLU',
    'Softmax',
]

__version__ = '0.2.0'
