"""
Train Decoder Transformer for Multi-Token Generation

This script trains a GPT-style decoder that can generate
multi-digit answers autoregressively.

Key features:
- Causal masking (can't see future)
- Next-token prediction at every position
- Autoregressive generation
- Works with double-digit numbers (0-99)
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import random
from datetime import datetime

from vocabluary import VOCAB_SIZE, VOCAB, REVERSE_VOCAB
from data_generatpr import generate_tokenized_dataset, split_dataset
from decoder_utils import prepare_decoder_batch, generate_autoregressive
from transformer_decoder import TransformerDecoder
from loss import CrossEntropyLoss
from optimizer import Adam
from evaluation import MetricsTracker, ModelCheckpoint


def compute_loss_and_accuracy(logits, targets, pad_token_id=0):
    """
    Compute loss and accuracy for next-token prediction.

    Only compute loss on non-padding tokens.

    Args:
        logits: Shape (batch, seq_len, vocab_size)
        targets: Shape (batch, seq_len)
        pad_token_id: Padding token ID to ignore

    Returns:
        tuple: (loss, accuracy)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch*seq_len,)

    # Create mask for non-padding tokens
    mask = (targets_flat != pad_token_id).astype(np.float32)
    num_valid = np.sum(mask)

    if num_valid == 0:
        return 0.0, 0.0

    # Compute loss (CrossEntropyLoss already averages over batch)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(logits_flat, targets_flat)

    # Compute accuracy (only on valid tokens)
    predictions = np.argmax(logits_flat, axis=1)
    correct = (predictions == targets_flat).astype(np.float32) * mask
    accuracy = np.sum(correct) / num_valid

    return loss, accuracy


def train_step(model, inputs, targets, loss_fn, optimizer, pad_token_id=0):
    """Single training step."""
    # Forward
    logits = model.forward(inputs)

    # Compute loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    loss = loss_fn.forward(logits_flat, targets_flat)

    # Backward
    grad = loss_fn.backward()
    grad = grad.reshape(batch_size, seq_len, vocab_size)
    model.backward(grad)

    # Update
    optimizer.step()

    # Compute metrics
    mask = (targets_flat != pad_token_id).astype(np.float32)
    num_valid = np.sum(mask)
    if num_valid > 0:
        predictions = np.argmax(logits_flat, axis=1)
        correct = (predictions == targets_flat).astype(np.float32) * mask
        accuracy = np.sum(correct) / num_valid
    else:
        accuracy = 0.0

    return loss, accuracy


def evaluate_generation(model, test_data, eos_token_id, num_examples=10):
    """
    Evaluate model by generating full sequences.

    Args:
        model: Trained decoder model
        test_data: List of (question_tokens, answer_tokens)
        eos_token_id: EOS token ID
        num_examples: Number of examples to test

    Returns:
        float: Accuracy (fraction of correct full sequences)
    """
    from vocabluary import detokenize

    correct = 0
    total = min(num_examples, len(test_data))

    for i in range(total):
        question_tokens, answer_tokens = test_data[i]

        # Generate
        generated = generate_autoregressive(
            model, question_tokens, eos_token_id, max_new_tokens=5, max_seq_len=20
        )

        # Check if correct
        if generated == answer_tokens:
            correct += 1

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 80)
    print("DECODER TRANSFORMER TRAINING - MULTI-TOKEN GENERATION")
    print("=" * 80)
    print()

    # Configuration
    dataset_size = 5000  # Smaller for faster training
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    max_value = 99

    print(f"Configuration:")
    print(f"  Dataset Size:   {dataset_size:,}")
    print(f"  Number Range:   0-{max_value}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Learning Rate:  {learning_rate}")
    print(f"  Model Type:     Decoder (GPT-style)")
    print()

    # Seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Generate data
    print("Generating dataset...")
    dataset = generate_tokenized_dataset(
        num_examples=dataset_size,
        num_args=3,
        max_value=max_value,
        balance_operations=True
    )
    train_data, val_data, test_data = split_dataset(dataset)
    print(f"✓ Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    print()

    # Prepare decoder batches
    print("Preparing decoder training data...")
    EOS_TOKEN = VOCAB['[EOS]']
    PAD_TOKEN = VOCAB['[PAD]']

    train_inputs, train_targets = prepare_decoder_batch(train_data, EOS_TOKEN, PAD_TOKEN, max_len=20)
    val_inputs, val_targets = prepare_decoder_batch(val_data, EOS_TOKEN, PAD_TOKEN, max_len=20)

    print(f"✓ Training sequences: {len(train_inputs):,}")
    print(f"  Input shape:  {train_inputs.shape}")
    print(f"  Target shape: {train_targets.shape}")
    print()

    # Create model
    print("Creating decoder model...")
    model = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=25
    )

    total_params = sum(p.size for p, _ in model.get_parameters())
    print(f"✓ Model created: {total_params:,} parameters")
    print()

    # Training setup
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=learning_rate)
    tracker = MetricsTracker()
    checkpoint = ModelCheckpoint('checkpoints/')

    # Training loop
    print("=" * 80)
    print("STARTING TRAINING")
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
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_inputs))

            batch_inputs = train_inputs_shuffled[start_idx:end_idx]
            batch_targets = train_targets_shuffled[start_idx:end_idx]

            loss, acc = train_step(model, batch_inputs, batch_targets, loss_fn, optimizer, PAD_TOKEN)

            epoch_loss += loss * len(batch_inputs)
            epoch_acc += acc * len(batch_inputs)

        train_loss = epoch_loss / len(train_inputs)
        train_acc = epoch_acc / len(train_inputs)

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        val_batches = (len(val_inputs) + batch_size - 1) // batch_size

        for batch_idx in range(val_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(val_inputs))

            batch_inputs = val_inputs[start_idx:end_idx]
            batch_targets = val_targets[start_idx:end_idx]

            logits = model.forward(batch_inputs)
            loss, acc = compute_loss_and_accuracy(logits, batch_targets, PAD_TOKEN)

            val_loss += loss * len(batch_inputs)
            val_acc += acc * len(batch_inputs)

        val_loss /= len(val_inputs)
        val_acc /= len(val_inputs)

        # Update tracker
        tracker.update(train_loss, train_acc, val_loss, val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

        # Save best
        if tracker.is_best_epoch():
            print(f"  🌟 New best validation accuracy!")
            checkpoint.save(model, optimizer, tracker, epoch, filename='best_decoder_model.pkl')

        print()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Final evaluation with generation
    print("=" * 80)
    print("TESTING AUTOREGRESSIVE GENERATION")
    print("=" * 80)
    print()

    gen_acc = evaluate_generation(model, test_data, EOS_TOKEN, num_examples=20)

    print(f"Generation Accuracy: {gen_acc:.2%} (20 examples)")
    print()

    # Show examples
    from vocabluary import detokenize

    print("Sample Generations:")
    print("-" * 80)

    for i in range(min(5, len(test_data))):
        question_tokens, answer_tokens = test_data[i]

        generated = generate_autoregressive(model, question_tokens, EOS_TOKEN, max_new_tokens=5, max_seq_len=20)

        question_str = ' '.join(detokenize(question_tokens))
        answer_str = ' '.join(detokenize(answer_tokens))
        generated_str = ' '.join(detokenize(generated)) if generated else "(empty)"

        match = "✓" if generated == answer_tokens else "✗"

        print(f"Q: {question_str}")
        print(f"A: {answer_str}")
        print(f"G: {generated_str} {match}")
        print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTraining time: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"Best Val Acc: {tracker.best_val_acc:.2%}")
    print(f"Generation Acc: {gen_acc:.2%}")
    print(f"\nModel saved to: checkpoints/best_decoder_model.pkl")


if __name__ == "__main__":
    main()
