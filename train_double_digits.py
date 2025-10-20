"""
Training Script for Double-Digit Numbers (0-99)

This script trains the transformer on numbers ranging from 0-99 instead of just 0-9.
Key differences from single-digit training:
- Numbers 0-99 (instead of 0-9)
- Dynamic sequence length calculation (no hardcoded max_length=20)
- Multi-digit outputs handled properly
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import random
from datetime import datetime

# Import components
from vocabluary import VOCAB_SIZE, VOCAB, REVERSE_VOCAB
from data_generatpr import generate_tokenized_dataset, split_dataset
from data_utils import create_batch, get_sequence_lengths
from transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from train_utils import train_step, evaluate
from evaluation import MetricsTracker, ModelCheckpoint


def calculate_max_length(dataset):
    """Calculate the maximum sequence length in the dataset."""
    max_len = max(len(example[0]) for example in dataset)
    # Add some buffer for safety
    return max_len + 5


def save_embeddings(model, epoch, filepath='embeddings_double_digits.txt', mode='a'):
    """
    Save current embedding vectors to file.

    Args:
        model: Transformer model
        epoch: Current epoch number (-1 for initial, 0+ for after epoch)
        filepath: Output file path
        mode: File mode ('w' for new file, 'a' for append)
    """
    embeddings = model.embedding.embeddings  # Shape: (vocab_size, embed_dim)

    with open(filepath, mode) as f:
        # Write header for this snapshot
        if epoch == -1:
            f.write("=" * 80 + "\n")
            f.write("INITIAL EMBEDDINGS (Before Training) - DOUBLE DIGITS (0-99)\n")
            f.write("=" * 80 + "\n")
        else:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"EMBEDDINGS AFTER EPOCH {epoch + 1}\n")
            f.write("=" * 80 + "\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape: {embeddings.shape}\n")
        f.write("\n")

        # Write each token's embedding
        for token_idx in range(VOCAB_SIZE):
            token_name = REVERSE_VOCAB[token_idx]
            embedding_vector = embeddings[token_idx]

            # Format: Token | Vector
            f.write(f"Token {token_idx:2d} ({token_name:8s}): ")

            # Write embedding vector with nice formatting
            vector_str = " ".join([f"{val:8.5f}" for val in embedding_vector])
            f.write(f"[{vector_str}]\n")

        f.write("\n")


def train_double_digits():
    """Main training function for double-digit numbers."""

    print("=" * 80)
    print("TRANSFORMER TRAINING - DOUBLE DIGITS (0-99)")
    print("=" * 80)
    print()

    # Configuration
    dataset_size = 10000
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    max_value = 99  # Train on 0-99 instead of 0-9

    print(f"Configuration:")
    print(f"  Dataset Size:   {dataset_size:,} examples")
    print(f"  Number Range:   0-{max_value} (double digits!)")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Learning Rate:  {learning_rate}")
    print()

    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random Seed: {seed}")
    print()

    # =========================================================================
    # Generate and Split Dataset
    # =========================================================================
    print("Generating dataset with double-digit numbers...")
    dataset = generate_tokenized_dataset(
        num_examples=dataset_size,
        num_args=3,
        max_value=max_value,  # 0-99
        balance_operations=True
    )
    print(f"✓ Generated {len(dataset):,} examples")

    # Show some statistics
    stats = get_sequence_lengths(dataset)
    print(f"\nSequence Length Statistics:")
    print(f"  Input min:  {stats['input_min']}")
    print(f"  Input max:  {stats['input_max']}")
    print(f"  Input mean: {stats['input_mean']:.1f}")
    print(f"  Answer min:  {stats['answer_min']}")
    print(f"  Answer max:  {stats['answer_max']}")
    print(f"  Answer mean: {stats['answer_mean']:.1f}")

    train_data, val_data, test_data = split_dataset(dataset)
    print(f"\n✓ Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    print()

    # =========================================================================
    # Calculate Max Length Dynamically
    # =========================================================================
    print("Calculating optimal max_length from dataset...")
    max_length = calculate_max_length(dataset)
    print(f"✓ Using max_length = {max_length} (dynamically calculated)")
    print()

    # =========================================================================
    # Prepare Batches
    # =========================================================================
    print("Preparing batches...")
    train_inputs, train_targets, _ = create_batch(train_data, max_length=max_length)
    train_targets = train_targets[:, 0]  # Take first digit only

    val_inputs, val_targets, _ = create_batch(val_data, max_length=max_length)
    val_targets = val_targets[:, 0]

    test_inputs, test_targets, _ = create_batch(test_data, max_length=max_length)
    test_targets = test_targets[:, 0]

    print(f"✓ Batches ready")
    print(f"  Train inputs shape: {train_inputs.shape}")
    print(f"  Val inputs shape:   {val_inputs.shape}")
    print(f"  Test inputs shape:  {test_inputs.shape}")
    print()

    # =========================================================================
    # Create Model
    # =========================================================================
    print("Creating model...")
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=max_length + 10  # Add buffer
    )

    params = model.get_parameters()
    total_params = sum(param.size for param, _ in params)
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"  Max sequence length: {model.pos_encoding.max_seq_len}")
    print()

    # =========================================================================
    # Save Initial Embeddings
    # =========================================================================
    print("Saving initial embeddings to embeddings_double_digits.txt...")
    save_embeddings(model, epoch=-1, filepath='embeddings_double_digits.txt', mode='w')
    print("✓ Initial embeddings saved")
    print()

    # =========================================================================
    # Training Setup
    # =========================================================================
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=learning_rate)
    tracker = MetricsTracker()
    checkpoint = ModelCheckpoint('checkpoints/')

    num_batches = (len(train_inputs) + batch_size - 1) // batch_size

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    start_time = datetime.now()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Shuffle training data
        shuffle_idx = np.random.permutation(len(train_inputs))
        train_inputs_shuffled = train_inputs[shuffle_idx]
        train_targets_shuffled = train_targets[shuffle_idx]

        # Train for one epoch
        epoch_loss = 0.0
        epoch_acc = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(train_inputs))

            batch_inputs = train_inputs_shuffled[start:end]
            batch_targets = train_targets_shuffled[start:end]

            loss, acc = train_step(model, batch_inputs, batch_targets, loss_fn, optimizer)
            epoch_loss += loss * len(batch_inputs)
            epoch_acc += acc * len(batch_inputs)

        # Compute epoch averages
        train_loss = epoch_loss / len(train_inputs)
        train_acc = epoch_acc / len(train_inputs)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_inputs, val_targets, loss_fn, batch_size=batch_size)

        # Update metrics
        tracker.update(train_loss, train_acc, val_loss, val_acc)

        # Print summary
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

        if tracker.is_best_epoch():
            print(f"  🌟 New best validation accuracy!")
            checkpoint.save(model, optimizer, tracker, epoch, filename='best_model_double_digits.pkl')

        # Save embeddings every 5 epochs and at the last epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"  💾 Saving embeddings after epoch {epoch + 1}...")
            save_embeddings(model, epoch, filepath='embeddings_double_digits.txt', mode='a')

        print()

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()

    print(f"Training time: {training_duration:.1f}s ({training_duration/60:.1f}min)")
    print()

    # Test set evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_inputs, test_targets, loss_fn, batch_size=batch_size)

    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {tracker.history['train_acc'][-1]:.2%}")
    print(f"  Val Accuracy:   {tracker.history['val_acc'][-1]:.2%}")
    print(f"  Test Accuracy:  {test_acc:.2%}")
    print(f"\nBest Performance:")
    print(f"  Best Val Acc:   {tracker.best_val_acc:.2%} (Epoch {tracker.best_epoch + 1})")
    print()

    # Save final embedding snapshot
    print("Saving final embeddings...")
    save_embeddings(model, epoch=num_epochs-1, filepath='embeddings_double_digits.txt', mode='a')
    print("✓ Final embeddings saved")
    print()

    print("=" * 80)
    print("Results saved to:")
    print("  - checkpoints/best_model_double_digits.pkl")
    print("  - embeddings_double_digits.txt")
    print("=" * 80)
    print()
    print("Note: This model can handle examples like Max(23, 45, 89)")
    print("      but not Max(212, ...) since trained range is 0-99")


if __name__ == "__main__":
    train_double_digits()
