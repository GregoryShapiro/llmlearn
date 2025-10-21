"""
Training Script for Double-Digit Numbers with Autoregressive Output

This script trains the transformer to generate multi-digit answers autoregressively:
- Input: Max(23, 45, 89)
- Output: 8 9 [EOS]  (generated token by token)

Key differences from previous version:
- Trains on full output sequences (not just first digit)
- Uses teacher forcing during training
- Model learns to generate until [EOS] token
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import random
from datetime import datetime

# Import components
from vocabluary import VOCAB_SIZE, VOCAB, REVERSE_VOCAB
from data_generatpr import generate_tokenized_dataset, split_dataset
from data_utils import get_sequence_lengths
from transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from evaluation import MetricsTracker, ModelCheckpoint


def calculate_max_length(dataset):
    """Calculate the maximum sequence length in the dataset."""
    max_len = max(len(example[0]) for example in dataset)
    return max_len + 5


def create_autoregressive_batches(data, max_input_len, max_output_len):
    """
    Create batches for autoregressive training.

    For each example:
    - Input: Max ( 2 3 , 4 5 , 8 9 )
    - Target output: 8 9 [EOS]

    We create multiple training examples with teacher forcing:
    - Input: Max(...) → Target: 8
    - Input: Max(...) 8 → Target: 9
    - Input: Max(...) 8 9 → Target: [EOS]

    Args:
        data: List of (input_tokens, answer_tokens) tuples
        max_input_len: Maximum input sequence length
        max_output_len: Maximum output sequence length

    Returns:
        inputs: Padded input sequences
        targets: Target tokens to predict
    """
    all_inputs = []
    all_targets = []

    EOS_TOKEN = VOCAB['[EOS]']
    PAD_TOKEN = VOCAB['[PAD]']

    for input_tokens, answer_tokens in data:
        # Add EOS to answer
        answer_with_eos = answer_tokens + [EOS_TOKEN]

        # For each position in the output, create a training example
        for i in range(len(answer_with_eos)):
            # Input: original question + answer tokens generated so far
            combined_input = input_tokens + answer_with_eos[:i]

            # Pad to max length
            if len(combined_input) < max_input_len:
                combined_input = combined_input + [PAD_TOKEN] * (max_input_len - len(combined_input))
            else:
                combined_input = combined_input[:max_input_len]

            # Target: next token in the answer
            target = answer_with_eos[i]

            all_inputs.append(combined_input)
            all_targets.append(target)

    return np.array(all_inputs), np.array(all_targets)


def predict_autoregressive(model, input_tokens, max_output_len=5):
    """
    Generate output autoregressively.

    Args:
        model: Trained transformer
        input_tokens: Input token indices (list)
        max_output_len: Maximum output length

    Returns:
        output_tokens: Generated token indices (list)
    """
    EOS_TOKEN = VOCAB['[EOS]']
    PAD_TOKEN = VOCAB['[PAD]']

    output_tokens = []
    current_input = input_tokens.copy()

    for _ in range(max_output_len):
        # Pad to model's expected length
        padded = current_input + [PAD_TOKEN] * (50 - len(current_input))
        padded = padded[:50]  # Truncate if too long

        # Predict next token
        input_batch = np.array(padded).reshape(1, -1)
        predictions = model.forward(input_batch)
        next_token = np.argmax(predictions[0])

        # Stop if EOS
        if next_token == EOS_TOKEN:
            break

        output_tokens.append(next_token)
        current_input.append(next_token)

    return output_tokens


def train_autoregressive():
    """Main training function with autoregressive output generation."""

    print("=" * 80)
    print("TRANSFORMER TRAINING - AUTOREGRESSIVE DOUBLE DIGITS (0-99)")
    print("=" * 80)
    print()

    # Configuration
    dataset_size = 10000
    num_epochs = 30  # More epochs for harder task
    batch_size = 32
    learning_rate = 0.001
    max_value = 99

    print(f"Configuration:")
    print(f"  Dataset Size:   {dataset_size:,} examples")
    print(f"  Number Range:   0-{max_value} (double digits!)")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Learning Rate:  {learning_rate}")
    print(f"  Training Mode:  Autoregressive (full sequence)")
    print()

    # Set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random Seed: {seed}")
    print()

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_tokenized_dataset(
        num_examples=dataset_size,
        num_args=3,
        max_value=max_value,
        balance_operations=True
    )
    print(f"✓ Generated {len(dataset):,} examples")

    stats = get_sequence_lengths(dataset)
    print(f"\nSequence Length Statistics:")
    print(f"  Input min:  {stats['input_min']}")
    print(f"  Input max:  {stats['input_max']}")
    print(f"  Answer min:  {stats['answer_min']}")
    print(f"  Answer max:  {stats['answer_max']}")

    train_data, val_data, test_data = split_dataset(dataset)
    print(f"\n✓ Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    print()

    # Create autoregressive batches
    print("Creating autoregressive training batches...")
    max_input_len = calculate_max_length(dataset)
    max_output_len = stats['answer_max'] + 1  # +1 for EOS

    print(f"  Max input length:  {max_input_len}")
    print(f"  Max output length: {max_output_len}")

    train_inputs, train_targets = create_autoregressive_batches(
        train_data, max_input_len, max_output_len
    )
    val_inputs, val_targets = create_autoregressive_batches(
        val_data, max_input_len, max_output_len
    )
    test_inputs, test_targets = create_autoregressive_batches(
        test_data, max_input_len, max_output_len
    )

    print(f"\n✓ Training examples: {len(train_inputs):,} (expanded from {len(train_data):,})")
    print(f"  Validation examples: {len(val_inputs):,}")
    print(f"  Test examples: {len(test_inputs):,}")
    print()

    # Create model
    print("Creating model...")
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=max_input_len + 10
    )

    params = model.get_parameters()
    total_params = sum(param.size for param, _ in params)
    print(f"✓ Model created with {total_params:,} parameters")
    print()

    # Training setup
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=learning_rate)
    tracker = MetricsTracker()
    checkpoint = ModelCheckpoint('checkpoints/')

    # Training loop
    print("=" * 80)
    print("STARTING AUTOREGRESSIVE TRAINING")
    print("=" * 80)
    print()

    start_time = datetime.now()
    num_batches = (len(train_inputs) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Shuffle
        shuffle_idx = np.random.permutation(len(train_inputs))
        train_inputs_shuffled = train_inputs[shuffle_idx]
        train_targets_shuffled = train_targets[shuffle_idx]

        # Train
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(train_inputs))

            batch_inputs = train_inputs_shuffled[start:end]
            batch_targets = train_targets_shuffled[start:end]

            # Forward pass
            predictions = model.forward(batch_inputs)

            # Compute loss
            loss = loss_fn.forward(predictions, batch_targets)

            # Backward pass
            grad = loss_fn.backward()
            model.backward(grad)

            # Update weights
            optimizer.step()

            # Track metrics
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == batch_targets)

            epoch_loss += loss * len(batch_inputs)
            epoch_acc += accuracy * len(batch_inputs)

        # Epoch averages
        train_loss = epoch_loss / len(train_inputs)
        train_acc = epoch_acc / len(train_inputs)

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        val_batches = (len(val_inputs) + batch_size - 1) // batch_size

        for batch_idx in range(val_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(val_inputs))

            batch_inputs = val_inputs[start:end]
            batch_targets = val_targets[start:end]

            predictions = model.forward(batch_inputs)
            loss = loss_fn.forward(predictions, batch_targets)

            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == batch_targets)

            val_loss += loss * len(batch_inputs)
            val_acc += accuracy * len(batch_inputs)

        val_loss /= len(val_inputs)
        val_acc /= len(val_inputs)

        # Update tracker
        tracker.update(train_loss, train_acc, val_loss, val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

        if tracker.is_best_epoch():
            print(f"  🌟 New best validation accuracy!")
            checkpoint.save(model, optimizer, tracker, epoch,
                          filename='best_model_double_digits_autoregressive.pkl')

        print()

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    # Final evaluation
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTraining time: {training_duration:.1f}s ({training_duration/60:.1f}min)")
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {tracker.history['train_acc'][-1]:.2%}")
    print(f"  Val Accuracy:   {tracker.history['val_acc'][-1]:.2%}")
    print(f"  Best Val Acc:   {tracker.best_val_acc:.2%} (Epoch {tracker.best_epoch + 1})")
    print()

    # Test autoregressive generation
    print("=" * 80)
    print("TESTING AUTOREGRESSIVE GENERATION")
    print("=" * 80)
    print()

    from vocabluary import detokenize

    for i in range(min(5, len(test_data))):
        input_tokens, answer_tokens = test_data[i]

        # Generate
        generated = predict_autoregressive(model, input_tokens, max_output_len=3)

        # Display
        input_str = ' '.join(detokenize(input_tokens))
        answer_str = ' '.join(detokenize(answer_tokens))
        generated_str = ' '.join(detokenize(generated))

        match = "✓" if generated == answer_tokens else "✗"

        print(f"Example {i+1}:")
        print(f"  Input:     {input_str}")
        print(f"  Expected:  {answer_str}")
        print(f"  Generated: {generated_str} {match}")
        print()

    print("=" * 80)
    print("Model saved to: checkpoints/best_model_double_digits_autoregressive.pkl")
    print("=" * 80)


if __name__ == "__main__":
    train_autoregressive()
