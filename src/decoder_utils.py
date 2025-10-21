"""
Decoder Utilities for Autoregressive Generation

This module provides utilities for building decoder-style transformers
that can generate sequences autoregressively (like GPT).
"""

import numpy as np


def create_causal_mask(seq_len):
    """
    Create a causal (lower triangular) attention mask.

    This prevents positions from attending to future positions,
    which is essential for autoregressive generation.

    Example for seq_len=4:
        [[1, 0, 0, 0],   ← Position 0 can only see position 0
         [1, 1, 0, 0],   ← Position 1 can see 0, 1
         [1, 1, 1, 0],   ← Position 2 can see 0, 1, 2
         [1, 1, 1, 1]]   ← Position 3 can see 0, 1, 2, 3

    Args:
        seq_len (int): Sequence length

    Returns:
        np.ndarray: Causal mask, shape (seq_len, seq_len)
                   1 = can attend, 0 = cannot attend
    """
    # Create lower triangular matrix
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def create_padding_mask(input_ids, pad_token_id=0):
    """
    Create a mask to prevent attending to padding tokens.

    Args:
        input_ids (np.ndarray): Input token IDs, shape (batch, seq_len)
        pad_token_id (int): ID of padding token

    Returns:
        np.ndarray: Padding mask, shape (batch, 1, 1, seq_len)
                   1 = real token (attend), 0 = padding (ignore)
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (input_ids != pad_token_id).astype(np.float32)

    # Expand dimensions for broadcasting
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    mask = mask[:, np.newaxis, np.newaxis, :]

    return mask


def combine_masks(causal_mask, padding_mask):
    """
    Combine causal mask and padding mask.

    Args:
        causal_mask (np.ndarray): Shape (seq_len, seq_len)
        padding_mask (np.ndarray): Shape (batch, 1, 1, seq_len)

    Returns:
        np.ndarray: Combined mask, shape (batch, 1, seq_len, seq_len)
    """
    # Expand causal mask to match batch dimension
    # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
    causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]

    # Broadcast and multiply
    # Both 1 = attend, either 0 = ignore
    combined = causal_mask * padding_mask

    return combined


def prepare_decoder_input(question_tokens, answer_tokens, eos_token_id, pad_token_id=0, max_len=50):
    """
    Prepare input for decoder training.

    Concatenates question + "=" + answer + EOS into single sequence.
    Creates training pairs for next-token prediction.

    Example:
        question: [Max, (, 2, 3, ,, 4, 5, ,, 8, 9, )]
        answer: [8, 9]

        Result:
        input_ids:  [Max, (, 2, 3, ,, 4, 5, ,, 8, 9, ), =, 8, 9]
        target_ids: [(, 2, 3, ,, 4, 5, ,, 8, 9, ), =, 8, 9, EOS]

    At each position, we predict the NEXT token.

    Args:
        question_tokens (list): Question token IDs
        answer_tokens (list): Answer token IDs
        eos_token_id (int): End-of-sequence token ID
        pad_token_id (int): Padding token ID
        max_len (int): Maximum sequence length

    Returns:
        tuple: (input_ids, target_ids)
            - input_ids: Tokens to feed to model
            - target_ids: Tokens to predict
    """
    # Combine: question + answer + EOS
    full_sequence = question_tokens + answer_tokens + [eos_token_id]

    # Input: all tokens except last
    # Target: all tokens except first
    input_ids = full_sequence[:-1]
    target_ids = full_sequence[1:]

    # Pad or truncate
    if len(input_ids) < max_len:
        # Pad
        padding_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_len
        target_ids = target_ids + [pad_token_id] * padding_len
    else:
        # Truncate
        input_ids = input_ids[:max_len]
        target_ids = target_ids[:max_len]

    return np.array(input_ids), np.array(target_ids)


def prepare_decoder_batch(data, eos_token_id, pad_token_id=0, max_len=50):
    """
    Prepare a batch of examples for decoder training.

    Args:
        data (list): List of (question_tokens, answer_tokens) tuples
        eos_token_id (int): End-of-sequence token ID
        pad_token_id (int): Padding token ID
        max_len (int): Maximum sequence length

    Returns:
        tuple: (input_batch, target_batch)
            - input_batch: Shape (batch_size, seq_len)
            - target_batch: Shape (batch_size, seq_len)
    """
    input_batch = []
    target_batch = []

    for question_tokens, answer_tokens in data:
        input_ids, target_ids = prepare_decoder_input(
            question_tokens, answer_tokens, eos_token_id, pad_token_id, max_len
        )
        input_batch.append(input_ids)
        target_batch.append(target_ids)

    return np.array(input_batch), np.array(target_batch)


def generate_autoregressive(model, prompt_tokens, eos_token_id, max_new_tokens=10, pad_token_id=0, max_seq_len=20):
    """
    Generate tokens autoregressively using the model.

    This is how real LLMs generate text - one token at a time,
    feeding each prediction back as input for the next.

    Args:
        model: Trained transformer model with forward() method
        prompt_tokens (list): Initial prompt token IDs
        eos_token_id (int): Stop generation when this token is predicted
        max_new_tokens (int): Maximum number of tokens to generate
        pad_token_id (int): Padding token ID
        max_seq_len (int): Maximum sequence length for model

    Returns:
        list: Generated token IDs (not including prompt)
    """
    generated = []
    current_tokens = prompt_tokens.copy()

    for _ in range(max_new_tokens):
        # Pad to model's expected length
        padded = current_tokens + [pad_token_id] * (max_seq_len - len(current_tokens))
        padded = padded[:max_seq_len]  # Truncate if too long

        # Predict next token
        input_batch = np.array(padded).reshape(1, -1)
        logits = model.forward(input_batch)  # Shape: (1, seq_len, vocab_size)

        # Get prediction for last non-padding position
        last_pos = min(len(current_tokens) - 1, logits.shape[1] - 1)
        next_token_logits = logits[0, last_pos, :]  # Shape: (vocab_size,)
        next_token = np.argmax(next_token_logits)

        # Stop if EOS
        if next_token == eos_token_id:
            break

        # Add to sequence
        generated.append(int(next_token))
        current_tokens.append(int(next_token))

    return generated
