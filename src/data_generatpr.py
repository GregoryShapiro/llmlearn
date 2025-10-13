"""
Data Generation Module

This module generates synthetic training data for the transformer model.
Each example consists of an operation (First, Second, Last, Max, Min) applied
to a list of numbers, along with the correct answer.

Initially focuses on 1-2 digit numbers for easier learning, but can be
extended to larger numbers for generalization testing.
"""

import random
from vocabluary import tokenize_with_numbers


# Define available operations and their implementations
OPERATIONS = {
    'First': lambda args: args[0],
    'Second': lambda args: args[1],
    'Last': lambda args: args[-1],
    'Max': lambda args: max(args),
    'Min': lambda args: min(args),
}


def generate_example(num_args=3, max_value=99, min_value=0, operations=None):
    """
    Generate a single training example with operation and answer.
    
    Args:
        num_args: Number of arguments for the operation (default: 3)
        max_value: Maximum value for random numbers (default: 99 for 1-2 digits)
        min_value: Minimum value for random numbers (default: 0)
        operations: List of operation names to choose from (default: all operations)
    
    Returns:
        Tuple of (input_sequence, answer):
            - input_sequence: List of tokens like ['Max', '(', 5, ',', 3, ',', 9, ')']
            - answer: Single integer representing the answer
    
    Example:
        >>> random.seed(42)
        >>> input_seq, answer = generate_example()
        >>> print(input_seq)
        ['Max', '(', 81, ',', 14, ',', 60, ')']
        >>> print(answer)
        81
    """
    # Select operation
    if operations is None:
        operations = list(OPERATIONS.keys())
    
    operation_name = random.choice(operations)
    operation_func = OPERATIONS[operation_name]
    
    # Generate random numbers as arguments
    numbers = [random.randint(min_value, max_value) for _ in range(num_args)]
    
    # Compute the correct answer
    answer = operation_func(numbers)
    
    # Build the input sequence: Operation ( num1 , num2 , num3 )
    input_sequence = [operation_name, '(']
    for i, num in enumerate(numbers):
        input_sequence.append(num)
        if i < len(numbers) - 1:
            input_sequence.append(',')
    input_sequence.append(')')
    
    return input_sequence, answer


def generate_tokenized_example(num_args=3, max_value=99, min_value=0, operations=None):
    """
    Generate a training example with tokenized input and output.
    
    This is a convenience function that generates an example and immediately
    tokenizes it for use in the model.
    
    Args:
        num_args: Number of arguments for the operation (default: 3)
        max_value: Maximum value for random numbers (default: 99 for 1-2 digits)
        min_value: Minimum value for random numbers (default: 0)
        operations: List of operation names to choose from (default: all operations)
    
    Returns:
        Tuple of (input_indices, answer_index):
            - input_indices: List of token indices for the input
            - answer_index: Single token index for the answer (or list for multi-digit)
    
    Example:
        >>> random.seed(42)
        >>> input_indices, answer_indices = generate_tokenized_example()
        >>> print(input_indices)
        [15, 17, 10, 3, 19, 3, 6, 19, 8, 2, 18]
        >>> print(answer_indices)
        [10, 3]
    """
    input_sequence, answer = generate_example(num_args, max_value, min_value, operations)
    
    # Tokenize input
    input_indices = tokenize_with_numbers(input_sequence)
    
    # Tokenize answer (multi-digit numbers become multiple tokens)
    answer_indices = tokenize_with_numbers([answer])
    
    return input_indices, answer_indices


def generate_dataset(num_examples, num_args=3, max_value=99, min_value=0, 
                     operations=None, balance_operations=True):
    """
    Generate a complete dataset of examples.
    
    Args:
        num_examples: Total number of examples to generate
        num_args: Number of arguments per operation (default: 3)
        max_value: Maximum value for random numbers (default: 99)
        min_value: Minimum value for random numbers (default: 0)
        operations: List of operation names to use (default: all)
        balance_operations: If True, ensure equal distribution of operations
    
    Returns:
        List of tuples (input_sequence, answer), each example in raw token form
    
    Example:
        >>> random.seed(42)
        >>> dataset = generate_dataset(10, max_value=9)
        >>> len(dataset)
        10
        >>> dataset[0]
        (['Min', '(', 8, ',', 1, ',', 6, ')'], 1)
    """
    dataset = []
    
    if operations is None:
        operations = list(OPERATIONS.keys())
    
    if balance_operations:
        # Generate equal number of examples for each operation
        examples_per_op = num_examples // len(operations)
        remainder = num_examples % len(operations)
        
        for op in operations:
            # Generate examples for this operation
            op_count = examples_per_op + (1 if remainder > 0 else 0)
            remainder -= 1
            
            for _ in range(op_count):
                example = generate_example(num_args, max_value, min_value, operations=[op])
                dataset.append(example)
    else:
        # Generate examples with random operation selection
        for _ in range(num_examples):
            example = generate_example(num_args, max_value, min_value, operations)
            dataset.append(example)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    return dataset


def generate_tokenized_dataset(num_examples, num_args=3, max_value=99, min_value=0,
                               operations=None, balance_operations=True):
    """
    Generate a complete dataset with tokenized examples.
    
    Args:
        num_examples: Total number of examples to generate
        num_args: Number of arguments per operation (default: 3)
        max_value: Maximum value for random numbers (default: 99)
        min_value: Minimum value for random numbers (default: 0)
        operations: List of operation names to use (default: all)
        balance_operations: If True, ensure equal distribution of operations
    
    Returns:
        List of tuples (input_indices, answer_indices)
    """
    raw_dataset = generate_dataset(num_examples, num_args, max_value, min_value,
                                   operations, balance_operations)
    
    tokenized_dataset = []
    for input_seq, answer in raw_dataset:
        input_indices = tokenize_with_numbers(input_seq)
        answer_indices = tokenize_with_numbers([answer])
        tokenized_dataset.append((input_indices, answer_indices))
    
    return tokenized_dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: List of examples
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        test_ratio: Proportion for testing (default: 0.1)
    
    Returns:
        Tuple of (train_set, val_set, test_set)
    
    Example:
        >>> dataset = generate_dataset(100)
        >>> train, val, test = split_dataset(dataset)
        >>> len(train), len(val), len(test)
        (80, 10, 10)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]
    
    return train_set, val_set, test_set


def get_operation_distribution(dataset):
    """
    Analyze the distribution of operations in a dataset.
    
    Args:
        dataset: List of (input_sequence, answer) tuples
    
    Returns:
        Dictionary mapping operation names to their counts
    
    Example:
        >>> dataset = generate_dataset(100, balance_operations=True)
        >>> dist = get_operation_distribution(dataset)
        >>> print(dist)
        {'First': 20, 'Second': 20, 'Last': 20, 'Max': 20, 'Min': 20}
    """
    distribution = {op: 0 for op in OPERATIONS.keys()}
    
    for input_seq, _ in dataset:
        # First token is the operation name
        operation = input_seq[0]
        if operation in distribution:
            distribution[operation] += 1
    
    return distribution


# ============================================================================
# Testing and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA GENERATION MODULE TEST")
    print("=" * 60)
    
    # Set seed for reproducibility in tests
    random.seed(42)
    
    # Test 1: Generate single examples
    print("\n1. Single Example Generation:")
    print("-" * 60)
    
    for i in range(5):
        input_seq, answer = generate_example()
        print(f"  Example {i+1}:")
        print(f"    Input:  {input_seq}")
        print(f"    Answer: {answer}")
        print()
    
    # Test 2: Generate examples with different parameters
    print("2. Examples with Different Parameters:")
    print("-" * 60)
    
    # Single digit only
    input_seq, answer = generate_example(num_args=3, max_value=9, min_value=0)
    print(f"  Single digit (0-9):")
    print(f"    Input:  {input_seq}")
    print(f"    Answer: {answer}")
    print()
    
    # Two digit numbers
    input_seq, answer = generate_example(num_args=3, max_value=99, min_value=10)
    print(f"  Two digits (10-99):")
    print(f"    Input:  {input_seq}")
    print(f"    Answer: {answer}")
    print()
    
    # More arguments
    input_seq, answer = generate_example(num_args=5, max_value=9)
    print(f"  Five arguments:")
    print(f"    Input:  {input_seq}")
    print(f"    Answer: {answer}")
    print()
    
    # Test 3: Tokenized examples
    print("3. Tokenized Examples:")
    print("-" * 60)
    
    for i in range(3):
        input_indices, answer_indices = generate_tokenized_example()
        print(f"  Example {i+1}:")
        print(f"    Input indices:  {input_indices}")
        print(f"    Answer indices: {answer_indices}")
        print()
    
    # Test 4: Generate dataset
    print("4. Dataset Generation:")
    print("-" * 60)
    
    dataset = generate_dataset(num_examples=20, max_value=9, balance_operations=True)
    print(f"  Generated {len(dataset)} examples")
    print(f"  Sample examples:")
    for i in range(3):
        input_seq, answer = dataset[i]
        print(f"    {i+1}. {input_seq} → {answer}")
    
    # Test 5: Operation distribution
    print("\n5. Operation Distribution Analysis:")
    print("-" * 60)
    
    dist = get_operation_distribution(dataset)
    print(f"  Distribution in dataset of {len(dataset)} examples:")
    for op, count in sorted(dist.items()):
        percentage = (count / len(dataset)) * 100
        print(f"    {op:8s}: {count:3d} ({percentage:5.1f}%)")
    
    # Test 6: Dataset splitting
    print("\n6. Dataset Splitting:")
    print("-" * 60)
    
    large_dataset = generate_dataset(num_examples=1000, max_value=99)
    train, val, test = split_dataset(large_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print(f"  Total examples: {len(large_dataset)}")
    print(f"  Train set:      {len(train)} ({len(train)/len(large_dataset)*100:.1f}%)")
    print(f"  Val set:        {len(val)} ({len(val)/len(large_dataset)*100:.1f}%)")
    print(f"  Test set:       {len(test)} ({len(test)/len(large_dataset)*100:.1f}%)")
    
    # Verify operation balance in splits
    print(f"\n  Operation distribution in train set:")
    train_dist = get_operation_distribution(train)
    for op, count in sorted(train_dist.items()):
        print(f"    {op:8s}: {count:3d}")
    
    # Test 7: Generate complete tokenized dataset
    print("\n7. Complete Tokenized Dataset:")
    print("-" * 60)
    
    tokenized_dataset = generate_tokenized_dataset(num_examples=10, max_value=9)
    print(f"  Generated {len(tokenized_dataset)} tokenized examples")
    print(f"  Sample tokenized examples:")
    for i in range(3):
        input_indices, answer_indices = tokenized_dataset[i]
        print(f"    {i+1}. Input:  {input_indices}")
        print(f"       Answer: {answer_indices}")
        print()
    
    # Test 8: Edge cases
    print("8. Edge Case Testing:")
    print("-" * 60)
    
    # All same numbers
    random.seed(123)
    examples_same = []
    for _ in range(10):
        input_seq, answer = generate_example(max_value=9)
        if len(set([x for x in input_seq if isinstance(x, int)])) == 1:
            examples_same.append((input_seq, answer))
    
    if examples_same:
        print(f"  Found examples with all same numbers:")
        for input_seq, answer in examples_same[:3]:
            print(f"    {input_seq} → {answer}")
    else:
        print(f"  Generated example with same numbers:")
        # Force it
        input_seq = ['Max', '(', 5, ',', 5, ',', 5, ')']
        answer = OPERATIONS['Max']([5, 5, 5])
        print(f"    {input_seq} → {answer}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nReady to generate training data!")
    print("Example usage:")
    print("  dataset = generate_dataset(10000, max_value=99)")
    print("  train, val, test = split_dataset(dataset)")
    