"""
Autoregressive Testing Script for Double-Digit Model

Tests the double-digit model with autoregressive generation.
Even though the model was only trained on first digits, we can try
to generate full sequences by feeding predictions back as input.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pickle

from vocabluary import VOCAB, REVERSE_VOCAB, tokenize_with_numbers, detokenize
from transformer import Transformer


def load_model(checkpoint_path='checkpoints/best_model_double_digits.pkl'):
    """Load trained model."""
    print(f"Loading model from {checkpoint_path}...")

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    model = Transformer(
        vocab_size=20,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=50
    )

    model_params = checkpoint['model_params']
    for i, (param, grad) in enumerate(model.get_parameters()):
        param[:] = model_params[f'param_{i}']

    print(f"✓ Model loaded (Val Acc: {checkpoint['best_val_acc']:.2%})")
    return model


def predict_autoregressive(model, operation, num1, num2, num3, max_digits=2):
    """
    Predict output autoregressively.

    Args:
        model: Trained model
        operation: Operation name
        num1, num2, num3: Input numbers (can be multi-digit)
        max_digits: Maximum digits to generate

    Returns:
        predicted_number: Integer result
        confidence: Average confidence
    """
    # Build input sequence (splits multi-digit numbers into digits)
    input_tokens = []
    input_tokens.append(operation)
    input_tokens.append('(')

    # Add first number (split digits)
    for digit in str(num1):
        input_tokens.append(digit)
    input_tokens.append(',')

    # Add second number
    for digit in str(num2):
        input_tokens.append(digit)
    input_tokens.append(',')

    # Add third number
    for digit in str(num3):
        input_tokens.append(digit)
    input_tokens.append(')')

    # Tokenize
    input_indices = tokenize_with_numbers(input_tokens)

    # Generate digits autoregressively
    generated_digits = []
    confidences = []
    current_input = input_indices.copy()

    for _ in range(max_digits):
        # Pad
        padded = current_input + [0] * (50 - len(current_input))
        padded = padded[:50]

        # Predict next digit
        input_batch = np.array(padded).reshape(1, -1)
        predictions = model.forward(input_batch)

        # Get predicted token
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]

        # Decode
        predicted_token = detokenize([predicted_idx])[0]

        # Check if it's a digit
        if predicted_token not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            break  # Stop if non-digit

        generated_digits.append(predicted_token)
        confidences.append(confidence)

        # Add to input for next prediction
        current_input.append(predicted_idx)

    # Convert to number
    if generated_digits:
        predicted_number = int(''.join(generated_digits))
        avg_confidence = np.mean(confidences)
    else:
        predicted_number = None
        avg_confidence = 0.0

    return predicted_number, avg_confidence, generated_digits


def test_examples():
    """Test the model with various examples."""
    model = load_model()

    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE TESTING - DOUBLE DIGITS")
    print("=" * 80)
    print()

    test_cases = [
        # Single digit (should work well)
        ('Max', 5, 8, 3, 8),
        ('Min', 7, 1, 5, 1),
        ('First', 3, 9, 2, 3),

        # Double digit (experimental - model wasn't trained for full sequences)
        ('Max', 23, 45, 89, 89),
        ('Min', 23, 45, 89, 23),
        ('First', 10, 20, 30, 10),
        ('Max', 12, 34, 56, 56),
    ]

    correct = 0
    for operation, n1, n2, n3, expected in test_cases:
        predicted, confidence, digits = predict_autoregressive(model, operation, n1, n2, n3)

        is_correct = (predicted == expected)
        if is_correct:
            correct += 1

        print(f"Test: {operation}({n1}, {n2}, {n3})")
        print(f"  Expected:   {expected}")
        print(f"  Predicted:  {predicted} (digits: {digits})")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Result:     {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print()

    print("=" * 80)
    print(f"Results: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 80)
    print()
    print("NOTE: This model was trained to predict only the FIRST digit.")
    print("Autoregressive generation is experimental and may not work well")
    print("for multi-digit outputs. For proper multi-digit support, retrain")
    print("with autoregressive training (train_double_digits_autoregressive.py).")


if __name__ == "__main__":
    test_examples()
