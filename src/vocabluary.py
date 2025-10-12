"""
Vocabulary and Tokenization Module

This module handles the conversion between text tokens and integer indices.
The vocabulary is designed for a simple operation language with digits.

Note: Numbers are tokenized at the CHARACTER level, so "25" becomes ['2', '5'].
This allows the model to generalize to any number length, though we'll train
primarily on 1-2 digit numbers initially.
"""

# Define the vocabulary: token -> index mapping
VOCAB = {
    # Special tokens
    '[PAD]': 0,   # Padding token for batching
    '[EOS]': 1,   # End of sequence token
    
    # Digits (0-9)
    '0': 2,
    '1': 3,
    '2': 4,
    '3': 5,
    '4': 6,
    '5': 7,
    '6': 8,
    '7': 9,
    '8': 10,
    '9': 11,
    
    # Operations
    'First': 12,
    'Second': 13,
    'Last': 14,
    'Max': 15,
    'Min': 16,
    
    # Syntax
    '(': 17,
    ')': 18,
    ',': 19,
}

# Create reverse vocabulary: index -> token mapping
REVERSE_VOCAB = {idx: token for token, idx in VOCAB.items()}

# Vocabulary size
VOCAB_SIZE = len(VOCAB)


def tokenize(sequence):
    """
    Convert a sequence of tokens (strings) to their integer indices.
    
    For multi-digit numbers, each digit should be a separate token.
    Use tokenize_with_numbers() for automatic number splitting.
    
    Args:
        sequence: List of string tokens, e.g., ['Max', '(', '5', ',', '3', ',', '9', ')']
                 or a single string token
    
    Returns:
        List of integer indices corresponding to the tokens
        
    Raises:
        KeyError: If a token is not in the vocabulary
    
    Example:
        >>> tokenize(['Max', '(', '5', ',', '3', ',', '9', ')'])
        [15, 17, 7, 19, 5, 19, 11, 18]
        
        >>> tokenize('Max')
        [15]
        
        >>> tokenize(['2', '5'])  # Multi-digit: each digit separate
        [4, 7]
    """
    # Handle single token (string) input
    if isinstance(sequence, str):
        if sequence not in VOCAB:
            raise KeyError(f"Token '{sequence}' not found in vocabulary")
        return [VOCAB[sequence]]
    
    # Handle list of tokens
    indices = []
    for token in sequence:
        if token not in VOCAB:
            raise KeyError(f"Token '{token}' not found in vocabulary")
        indices.append(VOCAB[token])
    
    return indices


def detokenize(indices):
    """
    Convert integer indices back to their string tokens.
    
    Args:
        indices: List of integer indices or a single integer index
    
    Returns:
        List of string tokens corresponding to the indices
        
    Raises:
        KeyError: If an index is not in the reverse vocabulary
    
    Example:
        >>> detokenize([15, 17, 7, 5, 11, 18])
        ['Max', '(', '5', '3', '9', ')']
        
        >>> detokenize(15)
        ['Max']
    """
    # Handle single index input
    if isinstance(indices, int):
        if indices not in REVERSE_VOCAB:
            raise KeyError(f"Index {indices} not found in reverse vocabulary")
        return [REVERSE_VOCAB[indices]]
    
    # Handle list of indices
    tokens = []
    for idx in indices:
        if idx not in REVERSE_VOCAB:
            raise KeyError(f"Index {idx} not found in reverse vocabulary")
        tokens.append(REVERSE_VOCAB[idx])
    
    return tokens


def get_token_index(token):
    """
    Get the index of a single token.
    
    Args:
        token: String token
        
    Returns:
        Integer index
        
    Example:
        >>> get_token_index('Max')
        15
    """
    if token not in VOCAB:
        raise KeyError(f"Token '{token}' not found in vocabulary")
    return VOCAB[token]


def get_token_from_index(index):
    """
    Get the token corresponding to an index.
    
    Args:
        index: Integer index
        
    Returns:
        String token
        
    Example:
        >>> get_token_from_index(15)
        'Max'
    """
    if index not in REVERSE_VOCAB:
        raise KeyError(f"Index {index} not found in reverse vocabulary")
    return REVERSE_VOCAB[index]


def is_valid_token(token):
    """
    Check if a token exists in the vocabulary.
    
    Args:
        token: String token to check
        
    Returns:
        Boolean indicating if token is valid
        
    Example:
        >>> is_valid_token('Max')
        True
        >>> is_valid_token('Invalid')
        False
    """
    return token in VOCAB


def is_valid_index(index):
    """
    Check if an index exists in the reverse vocabulary.
    
    Args:
        index: Integer index to check
        
    Returns:
        Boolean indicating if index is valid
        
    Example:
        >>> is_valid_index(15)
        True
        >>> is_valid_index(999)
        False
    """
    return index in REVERSE_VOCAB


def tokenize_with_numbers(sequence):
    """
    Tokenize a sequence, automatically splitting multi-digit numbers into individual digits.
    
    This is a convenience function that handles numbers (integers) by breaking them
    into individual digit characters before tokenization.
    
    Args:
        sequence: List of tokens where numbers can be integers or strings
                 e.g., ['Max', '(', 25, ',', 3, ',', 9, ')']
                 or ['Max', '(', '25', ',', '3', ',', '9', ')']
    
    Returns:
        List of integer indices with multi-digit numbers split
        
    Example:
        >>> tokenize_with_numbers(['Max', '(', 25, ',', 13, ',', 9, ')'])
        [15, 17, 4, 7, 19, 3, 5, 19, 11, 18]
        # 25 → ['2', '5'], 13 → ['1', '3'], 9 → ['9']
        
        >>> tokenize_with_numbers(['First', '(', 7, ',', 42, ',', 99, ')'])
        [12, 17, 9, 19, 6, 4, 19, 11, 11, 18]
        # 7 → ['7'], 42 → ['4', '2'], 99 → ['9', '9']
    """
    expanded_sequence = []
    
    for token in sequence:
        # Check if token is a number (int or numeric string)
        if isinstance(token, int) or (isinstance(token, str) and token.isdigit()):
            # Split multi-digit number into individual digits
            for digit in str(token):
                expanded_sequence.append(digit)
        else:
            # Keep non-numeric tokens as-is
            expanded_sequence.append(token)
    
    # Now tokenize the expanded sequence
    return tokenize(expanded_sequence)


def detokenize_with_numbers(indices, reconstruct_numbers=False):
    """
    Detokenize indices back to tokens, optionally reconstructing multi-digit numbers.
    
    Args:
        indices: List of integer indices
        reconstruct_numbers: If True, consecutive digit tokens are merged into multi-digit numbers
        
    Returns:
        List of string tokens
        
    Example:
        >>> detokenize_with_numbers([15, 17, 4, 7, 19, 11, 18])
        ['Max', '(', '2', '5', ',', '9', ')']
        
        >>> detokenize_with_numbers([15, 17, 4, 7, 19, 11, 18], reconstruct_numbers=True)
        ['Max', '(', '25', ',', '9', ')']
    """
    tokens = detokenize(indices)
    
    if not reconstruct_numbers:
        return tokens
    
    # Merge consecutive digits into numbers
    reconstructed = []
    current_number = ""
    
    for token in tokens:
        if token in '0123456789':
            # Accumulate digits
            current_number += token
        else:
            # Non-digit: flush accumulated number if any
            if current_number:
                reconstructed.append(current_number)
                current_number = ""
            reconstructed.append(token)
    
    # Don't forget the last number if sequence ends with a digit
    if current_number:
        reconstructed.append(current_number)
    
    return reconstructed


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VOCABULARY MODULE TEST")
    print("=" * 60)
    
    # Display vocabulary
    print("\n1. Complete Vocabulary:")
    print("-" * 60)
    for token, idx in sorted(VOCAB.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: '{token}'")
    
    print(f"\nVocabulary size: {VOCAB_SIZE}")
    
    # Test tokenization
    print("\n2. Tokenization Tests:")
    print("-" * 60)
    
    test_sequences = [
        ['Max', '(', '5', ',', '3', ',', '9', ')'],
        ['First', '(', '2', ',', '8', ',', '4', ')'],
        ['Min', '(', '7', ',', '1', ',', '9', ')'],
        ['Second', '(', '0', ',', '6', ',', '3', ')'],
        ['Last', '(', '1', ',', '4', ',', '8', ')'],
    ]
    
    for seq in test_sequences:
        indices = tokenize(seq)
        print(f"  Tokens:  {seq}")
        print(f"  Indices: {indices}")
        print()
    
    # Test detokenization
    print("3. Detokenization Tests:")
    print("-" * 60)
    
    test_indices = [
        [15, 17, 7, 19, 5, 19, 11, 18],
        [12, 17, 4, 19, 10, 19, 6, 18],
    ]
    
    for indices in test_indices:
        tokens = detokenize(indices)
        print(f"  Indices: {indices}")
        print(f"  Tokens:  {tokens}")
        print()
    
    # Test round-trip conversion
    print("4. Round-trip Conversion Test:")
    print("-" * 60)
    
    original = ['Max', '(', '5', ',', '3', ',', '9', ')']
    indices = tokenize(original)
    reconstructed = detokenize(indices)
    
    print(f"  Original:      {original}")
    print(f"  → Indices:     {indices}")
    print(f"  → Reconstructed: {reconstructed}")
    print(f"  Match: {original == reconstructed}")
    
    # Test single token operations
    print("\n5. Single Token Operations:")
    print("-" * 60)
    
    test_token = 'Max'
    test_index = 15
    
    print(f"  tokenize('{test_token}'): {tokenize(test_token)}")
    print(f"  detokenize({test_index}): {detokenize(test_index)}")
    print(f"  get_token_index('{test_token}'): {get_token_index(test_token)}")
    print(f"  get_token_from_index({test_index}): {get_token_from_index(test_index)}")
    
    # Test validation functions
    print("\n6. Validation Tests:")
    print("-" * 60)
    
    print(f"  is_valid_token('Max'): {is_valid_token('Max')}")
    print(f"  is_valid_token('Invalid'): {is_valid_token('Invalid')}")
    print(f"  is_valid_index(15): {is_valid_index(15)}")
    print(f"  is_valid_index(999): {is_valid_index(999)}")
    
    # Test error handling
    print("\n7. Error Handling Tests:")
    print("-" * 60)
    
    try:
        tokenize(['Invalid', 'Token'])
        print("  ERROR: Should have raised KeyError for invalid token")
    except KeyError as e:
        print(f"  ✓ Correctly raised KeyError: {e}")
    
    try:
        detokenize([999])
        print("  ERROR: Should have raised KeyError for invalid index")
    except KeyError as e:
        print(f"  ✓ Correctly raised KeyError: {e}")
    
    # Test multi-digit number tokenization
    print("\n8. Multi-Digit Number Tokenization:")
    print("-" * 60)
    
    test_multi_digit = [
        ['Max', '(', 25, ',', 13, ',', 9, ')'],
        ['First', '(', 7, ',', 42, ',', 99, ')'],
        ['Min', '(', 5, ',', 5, ',', 5, ')'],
    ]
    
    for seq in test_multi_digit:
        indices = tokenize_with_numbers(seq)
        tokens = detokenize(indices)
        reconstructed = detokenize_with_numbers(indices, reconstruct_numbers=True)
        
        print(f"  Original:        {seq}")
        print(f"  → Indices:       {indices}")
        print(f"  → As digits:     {tokens}")
        print(f"  → Reconstructed: {reconstructed}")
        print()
    
    # Test edge cases for number reconstruction
    print("9. Number Reconstruction Edge Cases:")
    print("-" * 60)
    
    # Single digit
    indices1 = tokenize_with_numbers(['Max', '(', 5, ',', 3, ',', 9, ')'])
    print(f"  Single digits: {detokenize_with_numbers(indices1, reconstruct_numbers=True)}")
    
    # Multi-digit
    indices2 = tokenize_with_numbers(['Max', '(', 50, ',', 13, ',', 99, ')'])
    print(f"  Multi-digits:  {detokenize_with_numbers(indices2, reconstruct_numbers=True)}")
    
    # Mixed
    indices3 = tokenize_with_numbers(['First', '(', 7, ',', 42, ',', 3, ')'])
    print(f"  Mixed:         {detokenize_with_numbers(indices3, reconstruct_numbers=True)}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    