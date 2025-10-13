"""
Step-by-Step Interactive Training Script

This script trains the transformer with detailed visualization at each step.
You'll see:
1. Dataset generation and statistics
2. Model architecture details
3. Forward pass internals (embeddings, attention, predictions)
4. Training progress with metrics
5. Per-operation analysis
6. Attention pattern visualization

Usage:
    python3 train_step_by_step.py --size small    # 1,000 examples
    python3 train_step_by_step.py --size medium   # 10,000 examples
    python3 train_step_by_step.py --size large    # 100,000 examples
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import random
import argparse
from datetime import datetime

# Import all components
from vocabluary import VOCAB_SIZE, VOCAB, tokenize_with_numbers, detokenize
from data_generatpr import generate_tokenized_dataset, split_dataset, get_operation_distribution
from data_utils import create_batch, get_sequence_lengths
from transformer import Transformer
from loss import CrossEntropyLoss, compute_accuracy
from optimizer import Adam
from train_utils import train_step, evaluate
from evaluation import MetricsTracker, ModelCheckpoint, compute_per_operation_accuracy, print_operation_analysis
from visualization import extract_attention_weights, visualize_attention_pattern, analyze_attention_patterns


def print_section(title, char="="):
    """Print a formatted section header."""
    width = 70
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def print_step(step_num, description):
    """Print a step marker."""
    print(f"\n{'─' * 70}")
    print(f"STEP {step_num}: {description}")
    print('─' * 70)


def decode_token(idx):
    """Helper to decode a single token index to string."""
    return detokenize([idx])[0]


def decode_tokens(indices):
    """Helper to decode a list of token indices to strings."""
    return [detokenize([idx])[0] for idx in indices]


def visualize_example(input_indices, answer_indices, example_num=1):
    """Visualize a single training example."""
    # Decode tokens
    input_tokens = decode_tokens(input_indices)
    answer_tokens = detokenize(answer_indices)
    answer_str = ''.join(answer_tokens) if all(t in '0123456789' for t in answer_tokens) else ' '.join(answer_tokens)

    print(f"\nExample {example_num}:")
    print(f"  Input:  {' '.join(input_tokens)}")
    print(f"  Answer: {answer_str}")
    print(f"  Token IDs: {input_indices}")


def show_model_architecture(model):
    """Display model architecture and parameter count."""
    print("\nModel Architecture:")
    print("-" * 70)

    params = model.get_parameters()
    total_params = sum(param.size for param, _ in params)

    print(f"  Vocabulary Size:     {model.vocab_size}")
    print(f"  Embedding Dimension: {model.embed_dim}")
    print(f"  Number of Layers:    {len(model.blocks)}")
    print(f"  Attention Heads:     {model.blocks[0].attention.num_heads}")
    # FFN hidden dim is the output size of linear1
    print(f"  FFN Hidden Dim:      {model.blocks[0].ffn.linear1.weights.shape[1]}")
    print(f"  Max Sequence Length: {model.pos_encoding.max_seq_len}")
    print(f"\n  Total Parameters:    {total_params:,}")

    # Parameter breakdown
    print(f"\n  Parameter Breakdown:")
    print(f"    Embedding:         {model.embedding.embeddings.size:,}")
    print(f"    Positional Enc:    {model.pos_encoding.encodings.size:,} (not trainable)")
    print(f"    Transformer Blocks: {sum(p.size for p, _ in model.blocks[0].get_parameters()) * len(model.blocks):,}")
    print(f"    Output Projection: {model.output_projection.weights.size + model.output_projection.bias.size:,}")


def show_forward_pass_details(model, example, verbose=True):
    """Show detailed forward pass for a single example."""
    input_indices, answer_indices = example

    # Prepare input
    max_len = 20
    input_padded = np.array(input_indices + [0] * (max_len - len(input_indices)))
    input_batch = input_padded.reshape(1, -1)

    if verbose:
        print("\nForward Pass Details:")
        print("-" * 70)
        visualize_example(input_indices, answer_indices)

    # Forward pass
    predictions = model.forward(input_batch)
    predicted_idx = np.argmax(predictions[0])
    predicted_token = decode_token(predicted_idx)
    correct_answer_tokens = detokenize(answer_indices)
    correct_answer = ''.join(correct_answer_tokens) if all(t in '0123456789' for t in correct_answer_tokens) else ' '.join(correct_answer_tokens)

    if verbose:
        print(f"\n  Forward Pass Steps:")
        print(f"    1. Embedding:     {input_batch.shape} → ({input_batch.shape[0]}, {input_batch.shape[1]}, {model.embed_dim})")
        print(f"    2. Pos Encoding:  Add position information")
        print(f"    3. Block 1:       Attention + FFN")
        print(f"    4. Block 2:       Attention + FFN")
        print(f"    5. Output Proj:   ({input_batch.shape[0]}, {input_batch.shape[1]}, {model.embed_dim}) → ({input_batch.shape[0]}, {model.vocab_size})")
        print(f"    6. Softmax:       Convert to probabilities")

        print(f"\n  Predictions:")
        print(f"    Predicted:  {predicted_token} (token ID: {predicted_idx})")
        print(f"    Correct:    {correct_answer} (token ID: {answer_indices[0]})")
        print(f"    Match:      {'✓ CORRECT' if predicted_idx == answer_indices[0] else '✗ WRONG'}")

        # Show top 5 predictions
        top5_indices = np.argsort(predictions[0])[::-1][:5]
        print(f"\n  Top 5 Predictions:")
        for i, idx in enumerate(top5_indices, 1):
            token = decode_token(idx)
            prob = predictions[0][idx]
            marker = "←" if idx == answer_indices[0] else ""
            print(f"    {i}. {token:4s}  {prob:6.2%}  {marker}")

    return predictions, predicted_idx == answer_indices[0]


def train_with_visualization(dataset_size, num_epochs=10, batch_size=32, learning_rate=0.001, interactive=True):
    """Train model with step-by-step visualization."""

    def wait_for_user(message="\nPress Enter to continue..."):
        """Wait for user input if in interactive mode."""
        if interactive:
            input(message)

    print_section("TRANSFORMER TRAINING - STEP BY STEP")

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # =========================================================================
    # STEP 1: Generate Dataset
    # =========================================================================
    print_step(1, "DATASET GENERATION")

    print(f"Generating {dataset_size:,} examples...")
    dataset = generate_tokenized_dataset(
        num_examples=dataset_size,
        num_args=3,
        max_value=9,
        balance_operations=True
    )
    print(f"✓ Generated {len(dataset):,} examples")

    # Show dataset statistics
    print("\nDataset Statistics:")
    print("-" * 70)

    # Operation distribution - count operation tokens manually
    from vocabluary import VOCAB
    operation_tokens = {'First': VOCAB['First'], 'Second': VOCAB['Second'],
                       'Last': VOCAB['Last'], 'Max': VOCAB['Max'], 'Min': VOCAB['Min']}
    operation_counts = {op: 0 for op in operation_tokens.keys()}

    for input_indices, _ in dataset:
        first_token = input_indices[0]
        for op_name, op_idx in operation_tokens.items():
            if first_token == op_idx:
                operation_counts[op_name] += 1
                break

    print("\nOperation Distribution:")
    for op, count in sorted(operation_counts.items()):
        pct = count / len(dataset) * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  {op:8s}: {bar} {pct:5.1f}%")

    # Sequence length statistics
    seq_stats = get_sequence_lengths(dataset)
    print(f"\nSequence Lengths:")
    print(f"  Input:  min={seq_stats['input_min']}, max={seq_stats['input_max']}, mean={seq_stats['input_mean']:.1f}")
    print(f"  Answer: min={seq_stats['answer_min']}, max={seq_stats['answer_max']}, mean={seq_stats['answer_mean']:.1f}")

    # Show sample examples
    print("\nSample Examples:")
    for i in range(min(3, len(dataset))):
        visualize_example(dataset[i][0], dataset[i][1], i+1)

    wait_for_user("\nPress Enter to continue to data splitting...")

    # =========================================================================
    # STEP 2: Split Dataset
    # =========================================================================
    print_step(2, "TRAIN/VAL/TEST SPLIT")

    train_data, val_data, test_data = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    print(f"Dataset Split:")
    print(f"  Train:      {len(train_data):,} examples ({len(train_data)/len(dataset)*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} examples ({len(val_data)/len(dataset)*100:.1f}%)")
    print(f"  Test:       {len(test_data):,} examples ({len(test_data)/len(dataset)*100:.1f}%)")

    wait_for_user("\nPress Enter to continue to data batching...")

    # =========================================================================
    # STEP 3: Prepare Batches
    # =========================================================================
    print_step(3, "DATA BATCHING AND PADDING")

    print("Creating batches with padding...")
    train_inputs, train_targets, train_masks = create_batch(train_data, max_length=20)
    train_targets = train_targets[:, 0]  # Take first digit only

    val_inputs, val_targets, val_masks = create_batch(val_data, max_length=20)
    val_targets = val_targets[:, 0]

    test_inputs, test_targets, test_masks = create_batch(test_data, max_length=20)
    test_targets = test_targets[:, 0]

    print(f"\nBatch Shapes:")
    print(f"  Train inputs:  {train_inputs.shape}  (batch_size, seq_len)")
    print(f"  Train targets: {train_targets.shape}  (batch_size,)")
    print(f"  Train masks:   {train_masks.shape}  (batch_size, seq_len)")

    print(f"\nExample Batch (first 3 examples):")
    for i in range(min(3, len(train_inputs))):
        tokens = decode_tokens(train_inputs[i][:8])  # Show first 8 tokens
        target = decode_token(train_targets[i])
        print(f"  {i+1}. Input: {' '.join(tokens)}... → Target: {target}")

    wait_for_user("\nPress Enter to continue to model creation...")

    # =========================================================================
    # STEP 4: Create Model
    # =========================================================================
    print_step(4, "MODEL ARCHITECTURE")

    print("Creating transformer model...")
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=50
    )
    print("✓ Model created")

    show_model_architecture(model)

    wait_for_user("\nPress Enter to see a forward pass example...")

    # =========================================================================
    # STEP 5: Forward Pass Example (Before Training)
    # =========================================================================
    print_step(5, "FORWARD PASS - BEFORE TRAINING")

    print("Running forward pass on a test example (untrained model)...")
    example = test_data[0]
    predictions, is_correct = show_forward_pass_details(model, example, verbose=True)

    print("\n💡 Note: Model is untrained, so predictions are random!")

    wait_for_user("\nPress Enter to continue to training setup...")

    # =========================================================================
    # STEP 6: Training Setup
    # =========================================================================
    print_step(6, "TRAINING SETUP")

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=learning_rate)
    tracker = MetricsTracker()
    checkpoint = ModelCheckpoint('checkpoints/')

    print("Training Configuration:")
    print(f"  Optimizer:      Adam")
    print(f"  Learning Rate:  {learning_rate}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Loss Function:  Cross-Entropy")

    num_batches = (len(train_inputs) + batch_size - 1) // batch_size
    print(f"\nTraining Details:")
    print(f"  Batches per epoch: {num_batches}")
    print(f"  Total updates:     {num_batches * num_epochs:,}")

    wait_for_user("\nPress Enter to start training...")

    # =========================================================================
    # STEP 7: Training Loop
    # =========================================================================
    print_step(7, "TRAINING LOOP")

    start_time = datetime.now()

    for epoch in range(num_epochs):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print('=' * 70)

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

            # Show progress every 10 batches or at the end
            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
                print(f"  Batch {batch_idx + 1:4d}/{num_batches} - Loss: {loss:.4f}, Acc: {acc:.4f}")

        # Compute epoch averages
        train_loss = epoch_loss / len(train_inputs)
        train_acc = epoch_acc / len(train_inputs)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_inputs, val_targets, loss_fn, batch_size=batch_size)

        # Update metrics
        tracker.update(train_loss, train_acc, val_loss, val_acc)

        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

        if tracker.is_best_epoch():
            print(f"  🌟 New best validation accuracy!")
            checkpoint.save(model, optimizer, tracker, epoch)

        # Show example predictions every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"\n  Example Prediction:")
            example = test_data[epoch % len(test_data)]
            _, is_correct = show_forward_pass_details(model, example, verbose=False)

            # Quick prediction display
            input_indices, answer_indices = example
            input_tokens = decode_tokens(input_indices)
            input_padded = np.array(input_indices + [0] * (20 - len(input_indices)))
            predictions = model.forward(input_padded.reshape(1, -1))
            predicted_idx = np.argmax(predictions[0])

            answer_tokens = detokenize(answer_indices)
            answer_str = ''.join(answer_tokens) if all(t in '0123456789' for t in answer_tokens) else ' '.join(answer_tokens)

            print(f"    Input:     {' '.join(input_tokens)}")
            print(f"    Expected:  {answer_str}")
            print(f"    Predicted: {decode_token(predicted_idx)} {'✓' if is_correct else '✗'}")

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total training time: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")

    tracker.print_summary()

    wait_for_user("\nPress Enter to evaluate on test set...")

    # =========================================================================
    # STEP 8: Test Set Evaluation
    # =========================================================================
    print_step(8, "TEST SET EVALUATION")

    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_inputs, test_targets, loss_fn, batch_size=batch_size)

    print(f"\nTest Set Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Per-operation analysis
    print_operation_analysis(model, test_data, ['First', 'Second', 'Last', 'Max', 'Min'])

    wait_for_user("\nPress Enter to see detailed predictions...")

    # =========================================================================
    # STEP 9: Detailed Predictions
    # =========================================================================
    print_step(9, "SAMPLE PREDICTIONS")

    print("Testing model on specific examples:\n")

    # Show predictions for each operation type
    operation_examples = {}
    for example in test_data:
        input_indices, _ = example
        op_token = input_indices[0]
        op_name = decode_token(op_token)

        if op_name in ['First', 'Second', 'Last', 'Max', 'Min']:
            if op_name not in operation_examples:
                operation_examples[op_name] = example

        if len(operation_examples) == 5:
            break

    for op_name in ['First', 'Second', 'Last', 'Max', 'Min']:
        if op_name in operation_examples:
            print(f"\n{op_name} Operation:")
            show_forward_pass_details(model, operation_examples[op_name], verbose=True)

    wait_for_user("\nPress Enter to see attention visualization...")

    # =========================================================================
    # STEP 10: Attention Visualization
    # =========================================================================
    print_step(10, "ATTENTION VISUALIZATION")

    print("Analyzing attention patterns...\n")

    # Show attention for one example
    example = test_data[0]
    input_indices, answer_indices = example

    # Prepare input
    input_padded = np.array(input_indices + [0] * (20 - len(input_indices)))
    input_batch = input_padded.reshape(1, -1)

    # Get token names
    token_names = decode_tokens(input_indices)

    # Extract attention
    attention_weights = extract_attention_weights(model, input_batch)

    # Visualize
    print(f"Example: {' '.join(token_names)}")
    visualize_attention_pattern(
        attention_weights,
        input_batch,
        token_names,
        head_idx=0,
        layer_idx=0,
        title="Attention Pattern - Layer 0, Head 0"
    )

    # Analyze pattern
    op_name = decode_token(input_indices[0])
    analysis = analyze_attention_patterns(attention_weights, input_batch[0], op_name)
    print(f"\nAttention Analysis:")
    print(f"  Pattern Type: {analysis['pattern_type']}")
    print(f"  Entropy:      {analysis['entropy']:.2f}")
    print(f"  (Low entropy = focused attention, High = distributed)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("TRAINING SUMMARY")

    print(f"Dataset Size:       {dataset_size:,} examples")
    print(f"Training Examples:  {len(train_data):,}")
    print(f"Epochs Trained:     {num_epochs}")
    print(f"Training Time:      {training_duration:.1f}s ({training_duration/60:.1f}min)")
    print(f"\nFinal Performance:")
    print(f"  Train Accuracy: {tracker.history['train_acc'][-1]:.2%}")
    print(f"  Val Accuracy:   {tracker.history['val_acc'][-1]:.2%}")
    print(f"  Test Accuracy:  {test_acc:.2%}")
    print(f"\nBest Performance:")
    print(f"  Best Val Acc:   {tracker.best_val_acc:.2%} (Epoch {tracker.best_epoch + 1})")

    print(f"\nCheckpoint saved: checkpoints/best_model.pkl")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train transformer with step-by-step visualization')
    parser.add_argument('--size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Dataset size: small (1K), medium (10K), large (100K)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Run without pausing for user input')

    args = parser.parse_args()

    # Map size to number of examples
    size_map = {
        'small': 1000,
        'medium': 10000,
        'large': 100000
    }

    dataset_size = size_map[args.size]

    print(f"\nConfiguration:")
    print(f"  Dataset Size: {dataset_size:,} examples ({args.size})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Interactive:  {'No' if args.no_interactive else 'Yes'}")

    if not args.no_interactive:
        input("\nPress Enter to begin...")

    train_with_visualization(
        dataset_size=dataset_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        interactive=not args.no_interactive
    )


if __name__ == "__main__":
    main()
