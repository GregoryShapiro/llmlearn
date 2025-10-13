"""
Data Utilities Module

This module handles batching, padding, and data preparation for the transformer model.
It converts variable-length sequences into fixed-length batches with appropriate
padding and attention masks.
"""

import numpy as np
from vocabluary import VOCAB


# Padding token index
PAD_TOKEN = VOCAB['[PAD]']


def pad_sequence(sequence, max_length, pad_value=PAD_TOKEN):
    """
    Pad a sequence to a fixed length.
    
    Args:
        sequence: List of token indices
        max_length: Target length to pad to
        pad_value: Value to use for padding (default: PAD_TOKEN)
    
    Returns:
        Padded sequence as numpy array
        
    Raises:
        ValueError: If sequence is longer than max_length
    
    Example:
        >>> pad_sequence([15, 17, 7, 19, 5], max_length=10)
        array([15, 17,  7, 19,  5,  0,  0,  0,  0,  0])
    """
    seq_len = len(sequence)
    
    if seq_len > max_length:
        raise ValueError(f"Sequence length {seq_len} exceeds max_length {max_length}")
    
    # Create padded array
    padded = np.full(max_length, pad_value, dtype=np.int32)
    padded[:seq_len] = sequence
    
    return padded


def create_attention_mask(sequence, max_length):
    """
    Create an attention mask for a sequence.
    
    The mask has 1s for real tokens and 0s for padding tokens.
    This tells the attention mechanism which positions to ignore.
    
    Args:
        sequence: List of token indices
        max_length: Length to pad the mask to
    
    Returns:
        Attention mask as numpy array (1 for real tokens, 0 for padding)
    
    Example:
        >>> create_attention_mask([15, 17, 7, 19, 5], max_length=10)
        array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    """
    seq_len = len(sequence)
    
    if seq_len > max_length:
        seq_len = max_length
    
    # Create mask: 1 for real tokens, 0 for padding
    mask = np.zeros(max_length, dtype=np.int32)
    mask[:seq_len] = 1
    
    return mask


def create_batch(examples, max_length=None, pad_value=PAD_TOKEN):
    """
    Create a batch from a list of examples with padding.
    
    Args:
        examples: List of tuples (input_indices, answer_indices)
        max_length: Maximum sequence length (if None, use longest in batch)
        pad_value: Value to use for padding
    
    Returns:
        Tuple of (input_batch, answer_batch, attention_masks):
            - input_batch: np.array of shape (batch_size, max_length)
            - answer_batch: np.array of shape (batch_size, max_answer_length)
            - attention_masks: np.array of shape (batch_size, max_length)
    
    Example:
        >>> examples = [
        ...     ([15, 17, 7, 19, 5, 19, 11, 18], [11]),
        ...     ([12, 17, 4, 19, 10, 18], [4]),
        ... ]
        >>> inputs, answers, masks = create_batch(examples)
        >>> inputs.shape
        (2, 8)
    """
    if not examples:
        raise ValueError("Cannot create batch from empty examples list")
    
    # Separate inputs and answers
    inputs = [ex[0] for ex in examples]
    answers = [ex[1] for ex in examples]
    
    # Determine max length if not provided
    if max_length is None:
        max_length = max(len(inp) for inp in inputs)
    
    # Determine max answer length
    max_answer_length = max(len(ans) for ans in answers)
    
    batch_size = len(examples)
    
    # Create padded batches
    input_batch = np.full((batch_size, max_length), pad_value, dtype=np.int32)
    answer_batch = np.full((batch_size, max_answer_length), pad_value, dtype=np.int32)
    attention_masks = np.zeros((batch_size, max_length), dtype=np.int32)
    
    # Fill in the data
    for i, (inp, ans) in enumerate(zip(inputs, answers)):
        inp_len = min(len(inp), max_length)
        ans_len = min(len(ans), max_answer_length)
        
        input_batch[i, :inp_len] = inp[:inp_len]
        answer_batch[i, :ans_len] = ans[:ans_len]
        attention_masks[i, :inp_len] = 1
    
    return input_batch, answer_batch, attention_masks


class DataLoader:
    """
    DataLoader for batching and iterating through a dataset.
    
    This class handles:
    - Batching examples
    - Shuffling data
    - Padding sequences
    - Creating attention masks
    """
    
    def __init__(self, dataset, batch_size, max_length=None, shuffle=True, pad_value=PAD_TOKEN):
        """
        Initialize the DataLoader.
        
        Args:
            dataset: List of (input_indices, answer_indices) tuples
            batch_size: Number of examples per batch
            max_length: Maximum sequence length (if None, computed from data)
            shuffle: Whether to shuffle data before each epoch
            pad_value: Value to use for padding
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_value = pad_value
        
        # Compute max_length if not provided
        if max_length is None:
            self.max_length = max(len(ex[0]) for ex in dataset)
        else:
            self.max_length = max_length
        
        # Compute max answer length
        self.max_answer_length = max(len(ex[1]) for ex in dataset)
        
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size
        self.indices = np.arange(len(dataset))
    
    def __len__(self):
        """Return the number of batches."""
        return self.num_batches
    
    def __iter__(self):
        """Iterate through batches."""
        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Yield batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            
            # Get examples for this batch
            batch_indices = self.indices[start_idx:end_idx]
            batch_examples = [self.dataset[idx] for idx in batch_indices]
            
            # Create padded batch
            input_batch, answer_batch, attention_masks = create_batch(
                batch_examples,
                max_length=self.max_length,
                pad_value=self.pad_value
            )
            
            yield input_batch, answer_batch, attention_masks
    
    def get_batch_stats(self):
        """
        Get statistics about the batches.
        
        Returns:
            Dictionary with batch statistics
        """
        return {
            'num_examples': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': self.num_batches,
            'max_length': self.max_length,
            'max_answer_length': self.max_answer_length,
        }


def get_sequence_lengths(dataset):
    """
    Analyze sequence lengths in a dataset.
    
    Args:
        dataset: List of (input_indices, answer_indices) tuples
    
    Returns:
        Dictionary with length statistics
    
    Example:
        >>> dataset = [([1, 2, 3], [1]), ([1, 2, 3, 4, 5], [2])]
        >>> stats = get_sequence_lengths(dataset)
        >>> stats['input_max']
        5
    """
    if not dataset:
        return {}
    
    input_lengths = [len(ex[0]) for ex in dataset]
    answer_lengths = [len(ex[1]) for ex in dataset]
    
    return {
        'input_min': min(input_lengths),
        'input_max': max(input_lengths),
        'input_mean': np.mean(input_lengths),
        'input_std': np.std(input_lengths),
        'answer_min': min(answer_lengths),
        'answer_max': max(answer_lengths),
        'answer_mean': np.mean(answer_lengths),
        'answer_std': np.std(answer_lengths),
    }


def calculate_padding_waste(dataset, max_length=None):
    """
    Calculate how much padding is wasted in a dataset.
    
    Args:
        dataset: List of (input_indices, answer_indices) tuples
        max_length: Maximum sequence length (if None, use longest sequence)
    
    Returns:
        Dictionary with padding statistics
    """
    if not dataset:
        return {}
    
    input_lengths = [len(ex[0]) for ex in dataset]
    
    if max_length is None:
        max_length = max(input_lengths)
    
    total_tokens = len(dataset) * max_length
    real_tokens = sum(input_lengths)
    padding_tokens = total_tokens - real_tokens
    padding_ratio = padding_tokens / total_tokens
    
    return {
        'max_length': max_length,
        'total_tokens': total_tokens,
        'real_tokens': real_tokens,
        'padding_tokens': padding_tokens,
        'padding_ratio': padding_ratio,
        'efficiency': 1.0 - padding_ratio,
    }


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA UTILS MODULE TEST")
    print("=" * 60)
    
    # Create some test examples
    test_examples = [
        ([15, 17, 7, 19, 5, 19, 11, 18], [11]),              # Max(5,3,9) = 9
        ([12, 17, 4, 19, 10, 19, 6, 18], [4]),               # First(2,8,4) = 2
        ([14, 17, 9, 19, 3, 19, 8, 18], [8]),                # Last(7,1,6) = 6
        ([15, 17, 7, 19, 11, 18], [11]),                     # Max(5,9) = 9 (shorter)
        ([16, 17, 4, 19, 7, 19, 3, 19, 11, 18], [3]),        # Min(2,5,1,9) = 1 (longer)
    ]
    
    # Test 1: Pad single sequence
    print("\n1. Padding Single Sequences:")
    print("-" * 60)
    
    seq = [15, 17, 7, 19, 5]
    padded = pad_sequence(seq, max_length=10)
    print(f"  Original:  {seq}")
    print(f"  Padded:    {padded}")
    print(f"  Shape:     {padded.shape}")
    
    # Test 2: Create attention mask
    print("\n2. Attention Mask Creation:")
    print("-" * 60)
    
    mask = create_attention_mask(seq, max_length=10)
    print(f"  Sequence:  {seq}")
    print(f"  Mask:      {mask}")
    print(f"  (1 = real token, 0 = padding)")
    
    # Test 3: Create batch
    print("\n3. Batch Creation:")
    print("-" * 60)
    
    input_batch, answer_batch, masks = create_batch(test_examples[:3])
    
    print(f"  Batch size: {input_batch.shape[0]}")
    print(f"  Max length: {input_batch.shape[1]}")
    print(f"\n  Input batch:")
    for i, inp in enumerate(input_batch):
        print(f"    Example {i+1}: {inp}")
    
    print(f"\n  Answer batch:")
    for i, ans in enumerate(answer_batch):
        print(f"    Example {i+1}: {ans}")
    
    print(f"\n  Attention masks:")
    for i, mask in enumerate(masks):
        print(f"    Example {i+1}: {mask}")
    
    # Test 4: Sequence length statistics
    print("\n4. Sequence Length Statistics:")
    print("-" * 60)
    
    stats = get_sequence_lengths(test_examples)
    print(f"  Input sequences:")
    print(f"    Min length:  {stats['input_min']}")
    print(f"    Max length:  {stats['input_max']}")
    print(f"    Mean length: {stats['input_mean']:.2f}")
    print(f"    Std dev:     {stats['input_std']:.2f}")
    print(f"\n  Answer sequences:")
    print(f"    Min length:  {stats['answer_min']}")
    print(f"    Max length:  {stats['answer_max']}")
    print(f"    Mean length: {stats['answer_mean']:.2f}")
    
    # Test 5: Padding waste analysis
    print("\n5. Padding Waste Analysis:")
    print("-" * 60)
    
    waste = calculate_padding_waste(test_examples)
    print(f"  Max length:      {waste['max_length']}")
    print(f"  Total tokens:    {waste['total_tokens']}")
    print(f"  Real tokens:     {waste['real_tokens']}")
    print(f"  Padding tokens:  {waste['padding_tokens']}")
    print(f"  Padding ratio:   {waste['padding_ratio']:.2%}")
    print(f"  Efficiency:      {waste['efficiency']:.2%}")
    
    # Test 6: DataLoader
    print("\n6. DataLoader Test:")
    print("-" * 60)
    
    # Create a larger test dataset
    import random
    random.seed(42)
    from data_generator import generate_tokenized_dataset
    
    dataset = generate_tokenized_dataset(num_examples=50, max_value=9)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size:   {dataloader.batch_size}")
    print(f"  Num batches:  {len(dataloader)}")
    
    stats = dataloader.get_batch_stats()
    print(f"\n  Batch statistics:")
    for key, value in stats.items():
        print(f"    {key:20s}: {value}")
    
    # Iterate through a few batches
    print(f"\n  First 3 batches:")
    for batch_idx, (inputs, answers, masks) in enumerate(dataloader):
        if batch_idx >= 3:
            break
        print(f"\n    Batch {batch_idx + 1}:")
        print(f"      Input shape:  {inputs.shape}")
        print(f"      Answer shape: {answers.shape}")
        print(f"      Mask shape:   {masks.shape}")
        print(f"      First example input:  {inputs[0]}")
        print(f"      First example answer: {answers[0]}")
        print(f"      First example mask:   {masks[0]}")
    
    # Test 7: Batch with different max_length
    print("\n7. Custom Max Length Test:")
    print("-" * 60)
    
    # Force a specific max length
    input_batch, answer_batch, masks = create_batch(test_examples[:3], max_length=15)
    print(f"  Forced max_length: 15")
    print(f"  Resulting batch shape: {input_batch.shape}")
    print(f"  Example with padding:")
    print(f"    Input: {input_batch[0]}")
    print(f"    Mask:  {masks[0]}")
    
    # Test 8: Edge cases
    print("\n8. Edge Case Testing:")
    print("-" * 60)
    
    # Single example
    single_batch = create_batch([test_examples[0]])
    print(f"  Single example batch shape: {single_batch[0].shape}")
    
    # All same length
    same_length_examples = [
        ([1, 2, 3, 4, 5], [1]),
        ([6, 7, 8, 9, 10], [2]),
        ([11, 12, 13, 14, 15], [3]),
    ]
    same_batch, _, same_masks = create_batch(same_length_examples)
    print(f"  Same length examples - padding needed: {(same_batch == PAD_TOKEN).any()}")
    
    # Test error handling
    print(f"\n  Testing error handling:")
    try:
        # Try to pad sequence longer than max_length
        pad_sequence([1, 2, 3, 4, 5], max_length=3)
        print("    ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"    ✓ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nDataLoader is ready for training!")
    print("Example usage:")
    print("  from data_utils import DataLoader")
    print("  dataloader = DataLoader(train_data, batch_size=32)")
    print("  for inputs, answers, masks in dataloader:")
    print("      # Train on batch")
    print("      pass")
    