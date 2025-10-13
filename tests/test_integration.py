"""
End-to-End Integration Test

This test verifies that all components work together correctly:
1. Data generation and tokenization
2. Model forward pass
3. Loss computation
4. Backward pass and gradient flow
5. Optimizer updates
6. Training loop

This is a critical test that ensures the complete pipeline is functional
before attempting full-scale training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import random

# Import all components
from vocabluary import tokenize_with_numbers, VOCAB_SIZE
from data_generatpr import generate_example, generate_dataset, split_dataset
from data_utils import create_batch
from transformer import Transformer
from loss import CrossEntropyLoss, compute_accuracy
from optimizer import Adam
from train_utils import train_step, evaluate


def test_data_generation():
    """Test that data generation works correctly."""
    print("Testing data generation...")

    # Generate a single example
    input_seq, answer = generate_example(num_args=3, max_value=9)

    # Verify structure
    assert isinstance(input_seq, list), "Input should be a list"
    assert isinstance(answer, int), "Answer should be an integer"
    assert len(input_seq) >= 5, "Input should have at least 5 tokens"

    # Check that operation is first token
    assert input_seq[0] in ['First', 'Second', 'Last', 'Max', 'Min'], \
        "First token should be an operation"

    print(f"  Sample: {input_seq} → {answer}")
    print("✓ Data generation works")


def test_tokenization():
    """Test that tokenization works correctly."""
    print("Testing tokenization...")

    # Generate and tokenize example
    input_seq, answer = generate_example(num_args=3, max_value=9)
    input_indices = tokenize_with_numbers(input_seq)
    answer_indices = tokenize_with_numbers([answer])

    # Verify indices are valid
    assert all(0 <= idx < VOCAB_SIZE for idx in input_indices), \
        "All input indices should be in vocabulary"
    assert all(0 <= idx < VOCAB_SIZE for idx in answer_indices), \
        "All answer indices should be in vocabulary"

    print(f"  Tokenized input:  {input_indices}")
    print(f"  Tokenized answer: {answer_indices}")
    print("✓ Tokenization works")


def test_batching():
    """Test that batching and padding work correctly."""
    print("Testing batching...")

    # Generate multiple examples
    examples = []
    for _ in range(5):
        input_seq, answer = generate_example(num_args=3, max_value=9)
        input_indices = tokenize_with_numbers(input_seq)
        answer_indices = tokenize_with_numbers([answer])
        # For our toy problem, we treat answer as a classification
        # Use the first digit as the target class
        target = answer_indices[0]
        examples.append((input_indices, [target]))

    # Create batch
    batch_tokens, batch_targets, _ = create_batch(examples, max_length=20)
    batch_targets = batch_targets[:, 0]  # Take first digit only

    # Verify shapes
    assert batch_tokens.shape == (5, 20), \
        f"Expected batch shape (5, 20), got {batch_tokens.shape}"
    assert batch_targets.shape == (5,), \
        f"Expected targets shape (5,), got {batch_targets.shape}"

    print(f"  Batch tokens shape:  {batch_tokens.shape}")
    print(f"  Batch targets shape: {batch_targets.shape}")
    print("✓ Batching works")


def test_model_forward_pass():
    """Test that model forward pass produces correct output."""
    print("Testing model forward pass...")

    # Create small model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )

    # Create dummy input
    batch_tokens = np.array([[15, 17, 7, 5, 11, 18, 0, 0]])  # Padded sequence

    # Forward pass
    predictions = model.forward(batch_tokens)

    # Verify output shape
    assert predictions.shape == (1, VOCAB_SIZE), \
        f"Expected output shape (1, {VOCAB_SIZE}), got {predictions.shape}"

    # Verify predictions are probabilities
    assert np.allclose(np.sum(predictions, axis=1), 1.0), \
        "Predictions should sum to 1 (probabilities)"
    assert np.all(predictions >= 0) and np.all(predictions <= 1), \
        "Predictions should be in [0, 1]"

    print(f"  Output shape: {predictions.shape}")
    print(f"  Output sums to 1: {np.allclose(np.sum(predictions), 1.0)}")
    print("✓ Model forward pass works")


def test_loss_computation():
    """Test that loss computation works with model output."""
    print("Testing loss computation...")

    # Create model and loss
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()

    # Generate example
    input_seq, answer = generate_example(num_args=3, max_value=9)
    input_indices = tokenize_with_numbers(input_seq)
    answer_indices = tokenize_with_numbers([answer])
    target = answer_indices[0]  # First digit as target

    # Prepare batch
    batch_tokens = np.array([input_indices + [0] * (20 - len(input_indices))])
    batch_targets = np.array([target])

    # Forward pass
    predictions = model.forward(batch_tokens)

    # Compute loss
    loss = loss_fn.forward(predictions, batch_targets)

    # Verify loss is reasonable
    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"
    assert not np.isnan(loss), "Loss should not be NaN"
    assert not np.isinf(loss), "Loss should not be Inf"

    print(f"  Loss value: {loss:.4f}")
    print("✓ Loss computation works")


def test_backward_pass():
    """Test that backward pass computes gradients."""
    print("Testing backward pass...")

    # Create model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()

    # Create dummy data
    batch_tokens = np.array([[15, 17, 7, 5, 11, 18, 0, 0]])
    batch_targets = np.array([7])  # Token '5'

    # Forward and backward
    predictions = model.forward(batch_tokens)
    loss = loss_fn.forward(predictions, batch_targets)
    grad_output = loss_fn.backward()
    model.backward(grad_output)

    # Check that gradients were computed
    params = model.get_parameters()
    gradients_computed = sum(1 for _, grad in params if grad is not None)

    assert gradients_computed > 0, "Some parameters should have gradients"

    print(f"  Total parameters: {len(params)}")
    print(f"  Parameters with gradients: {gradients_computed}")
    print("✓ Backward pass works")


def test_optimizer_update():
    """Test that optimizer updates parameters."""
    print("Testing optimizer update...")

    # Create model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=0.001)

    # Get initial parameter values
    params = model.get_parameters()
    initial_values = [param.copy() for param, _ in params]

    # Create dummy data
    batch_tokens = np.array([[15, 17, 7, 5, 11, 18, 0, 0]])
    batch_targets = np.array([7])

    # Training step
    model.zero_grad()
    predictions = model.forward(batch_tokens)
    loss = loss_fn.forward(predictions, batch_targets)
    grad_output = loss_fn.backward()
    model.backward(grad_output)
    optimizer.step()

    # Check that parameters changed (get fresh parameter references)
    params_after = model.get_parameters()
    params_changed = 0
    for initial, (current, _) in zip(initial_values, params_after):
        if not np.allclose(initial, current):
            params_changed += 1

    assert params_changed > 0, f"Some parameters should have changed (0/{len(params_after)} changed)"

    print(f"  Parameters changed: {params_changed}/{len(params_after)}")
    print("✓ Optimizer update works")


def test_training_step():
    """Test that a single training step works end-to-end."""
    print("Testing complete training step...")

    # Create model, loss, optimizer
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=0.001)

    # Generate batch
    examples = []
    for _ in range(4):
        input_seq, answer = generate_example(num_args=3, max_value=9)
        input_indices = tokenize_with_numbers(input_seq)
        answer_indices = tokenize_with_numbers([answer])
        target = answer_indices[0]
        examples.append((input_indices, [target]))

    batch_tokens, batch_targets, _ = create_batch(examples, max_length=20)
    batch_targets = batch_targets[:, 0]  # Take first digit only

    # Perform training step
    loss, accuracy = train_step(model, batch_tokens, batch_targets, loss_fn, optimizer)

    # Verify metrics
    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"
    assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"

    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print("✓ Training step works")


def test_multiple_training_steps():
    """Test that loss decreases over multiple training steps."""
    print("Testing multiple training steps...")

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create model (smaller for faster testing)
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.get_parameters, learning_rate=0.01)  # Higher lr for faster learning

    # Generate small training set (same examples repeated for overfitting test)
    examples = []
    for _ in range(10):
        input_seq, answer = generate_example(num_args=3, max_value=9)
        input_indices = tokenize_with_numbers(input_seq)
        answer_indices = tokenize_with_numbers([answer])
        target = answer_indices[0]
        examples.append((input_indices, [target]))

    # Train for multiple steps
    losses = []
    for step in range(20):
        # Create batch
        batch_tokens, batch_targets, _ = create_batch(examples, max_length=20)
        batch_targets = batch_targets[:, 0]  # Take first digit only

        # Training step
        loss, accuracy = train_step(model, batch_tokens, batch_targets, loss_fn, optimizer)
        losses.append(loss)

        if step % 5 == 0:
            print(f"  Step {step}: Loss = {loss:.4f}, Acc = {accuracy:.4f}")

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Improvement:  {initial_loss - final_loss:.4f}")

    # Loss should decrease (model should be able to overfit on small dataset)
    assert final_loss < initial_loss, \
        f"Loss should decrease with training (initial: {initial_loss:.4f}, final: {final_loss:.4f})"

    print("✓ Model learns (loss decreases)")


def test_evaluation_function():
    """Test that evaluation function works correctly."""
    print("Testing evaluation function...")

    # Create model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64,
        max_seq_len=20
    )
    loss_fn = CrossEntropyLoss()

    # Generate validation set
    examples = []
    for _ in range(10):
        input_seq, answer = generate_example(num_args=3, max_value=9)
        input_indices = tokenize_with_numbers(input_seq)
        answer_indices = tokenize_with_numbers([answer])
        target = answer_indices[0]
        examples.append((input_indices, [target]))

    # Prepare data for evaluate function
    val_tokens, val_targets, _ = create_batch(examples, max_length=20)
    val_targets = val_targets[:, 0]  # Take first digit only

    # Evaluate
    val_loss, val_accuracy = evaluate(model, val_tokens, val_targets, loss_fn, batch_size=5)

    # Verify results
    assert isinstance(val_loss, (float, np.floating)), "Loss should be a scalar"
    assert 0 <= val_accuracy <= 1, "Accuracy should be in [0, 1]"

    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Validation accuracy: {val_accuracy:.4f}")
    print("✓ Evaluation function works")


# Run all tests
if __name__ == '__main__':
    print("\n" + "="*70)
    print("RUNNING END-TO-END INTEGRATION TESTS")
    print("="*70 + "\n")

    try:
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        print("Phase 1: Data Pipeline")
        print("-" * 70)
        test_data_generation()
        test_tokenization()
        test_batching()

        print("\nPhase 2-4: Model Architecture")
        print("-" * 70)
        test_model_forward_pass()

        print("\nPhase 5: Training Infrastructure")
        print("-" * 70)
        test_loss_computation()
        test_backward_pass()
        test_optimizer_update()

        print("\nEnd-to-End Integration")
        print("-" * 70)
        test_training_step()
        test_multiple_training_steps()
        test_evaluation_function()

        print("\n" + "="*70)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("="*70)
        print("\nThe transformer is fully functional and ready for training!")
        print("All components work together correctly:")
        print("  ✓ Data generation and tokenization")
        print("  ✓ Model forward pass")
        print("  ✓ Loss computation")
        print("  ✓ Backward pass and gradients")
        print("  ✓ Optimizer updates")
        print("  ✓ Training loop")
        print("  ✓ Model learns (loss decreases)")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise
