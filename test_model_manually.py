"""
Manual Model Testing Script

This script loads a trained model and lets you test it interactively or with custom examples.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pickle

from vocabluary import VOCAB, REVERSE_VOCAB, tokenize_with_numbers, detokenize
from transformer import Transformer

def load_trained_model(checkpoint_path='checkpoints/best_model_run2.pkl'):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Create model with same architecture
    model = Transformer(
        vocab_size=20,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=50
    )
    
    # Load trained parameters
    model_params = checkpoint['model_params']
    saved_params = model.get_parameters()

    for i, (param, grad) in enumerate(saved_params):
        param[:] = model_params[f'param_{i}']
    
    print(f"✓ Model loaded successfully!")
    print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2%}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    
    return model

def predict(model, operation, num1, num2, num3):
    """
    Make a prediction for a given operation and numbers.
    
    Args:
        model: Trained transformer model
        operation: One of 'First', 'Second', 'Last', 'Max', 'Min'
        num1, num2, num3: Numbers (0-9)
    
    Returns:
        Predicted number and confidence
    """
    # Build input sequence
    input_sequence = [operation, '(', num1, ',', num2, ',', num3, ')']
    
    # Tokenize
    input_indices = tokenize_with_numbers(input_sequence)
    
    # Pad to max length (20)
    input_padded = input_indices + [0] * (20 - len(input_indices))
    input_batch = np.array(input_padded).reshape(1, -1)
    
    # Forward pass
    predictions = model.forward(input_batch)
    
    # Get prediction
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    # Decode
    predicted_token = detokenize([predicted_idx])[0]
    
    return predicted_token, confidence, predictions[0]

def print_detailed_prediction(model, operation, num1, num2, num3, correct_answer=None):
    """Print detailed prediction with top-5 results."""
    print("\n" + "=" * 70)
    print(f"Testing: {operation}({num1}, {num2}, {num3})")
    print("=" * 70)
    
    predicted, confidence, all_probs = predict(model, operation, num1, num2, num3)
    
    # Compute correct answer if not provided
    if correct_answer is None:
        if operation == 'First':
            correct_answer = num1
        elif operation == 'Second':
            correct_answer = num2
        elif operation == 'Last':
            correct_answer = num3
        elif operation == 'Max':
            correct_answer = max(num1, num2, num3)
        elif operation == 'Min':
            correct_answer = min(num1, num2, num3)
    
    # Check if correct
    is_correct = (str(predicted) == str(correct_answer))
    
    print(f"\nPredicted: {predicted}")
    print(f"Correct:   {correct_answer}")
    print(f"Result:    {'✓ CORRECT' if is_correct else '✗ WRONG'}")
    print(f"Confidence: {confidence:.2%}")
    
    # Show top 5 predictions
    print(f"\nTop 5 Predictions:")
    top5_indices = np.argsort(all_probs)[::-1][:5]
    for i, idx in enumerate(top5_indices, 1):
        token = detokenize([idx])[0]
        prob = all_probs[idx]
        marker = "←" if idx == VOCAB.get(str(correct_answer), -1) else ""
        print(f"  {i}. {token:8s}  {prob:6.2%}  {marker}")
    
    return is_correct

def interactive_mode(model):
    """Interactive testing mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nTest your model interactively!")
    print("Operations: First, Second, Last, Max, Min")
    print("Numbers: 0-9")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get operation
            operation = input("Operation (First/Second/Last/Max/Min): ").strip()
            if operation.lower() == 'quit':
                break
            
            if operation not in ['First', 'Second', 'Last', 'Max', 'Min']:
                print("Invalid operation! Use: First, Second, Last, Max, or Min")
                continue
            
            # Get numbers
            num1 = int(input("Number 1 (0-9): "))
            num2 = int(input("Number 2 (0-9): "))
            num3 = int(input("Number 3 (0-9): "))
            
            if not all(0 <= n <= 9 for n in [num1, num2, num3]):
                print("Numbers must be between 0 and 9!")
                continue
            
            # Make prediction
            print_detailed_prediction(model, operation, num1, num2, num3)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def batch_test_mode(model):
    """Test multiple examples at once."""
    print("\n" + "=" * 70)
    print("BATCH TEST MODE")
    print("=" * 70)
    
    # Test cases covering different scenarios
    test_cases = [
        # Easy cases
        ('First', 5, 3, 8),
        ('Second', 2, 7, 1),
        ('Last', 4, 6, 9),
        ('Max', 3, 8, 2),
        ('Min', 7, 1, 5),
        
        # Edge cases - all same
        ('Max', 5, 5, 5),
        ('Min', 3, 3, 3),
        
        # Edge cases - duplicates
        ('Max', 9, 2, 9),
        ('Min', 1, 8, 1),
        ('Second', 4, 4, 7),
        
        # Edge cases - extreme values
        ('Max', 0, 0, 9),
        ('Min', 9, 0, 9),
        ('First', 0, 5, 9),
        ('Last', 1, 4, 0),
    ]
    
    print(f"\nTesting {len(test_cases)} examples...\n")
    
    correct = 0
    for operation, n1, n2, n3 in test_cases:
        is_correct = print_detailed_prediction(model, operation, n1, n2, n3)
        if is_correct:
            correct += 1
        input("\nPress Enter for next example...")
    
    print("\n" + "=" * 70)
    print(f"BATCH TEST RESULTS: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 70)

def main():
    """Main entry point."""
    print("=" * 70)
    print("TRAINED MODEL MANUAL TESTING")
    print("=" * 70)
    
    # Load model
    model = load_trained_model('checkpoints/best_model_run2.pkl')
    
    print("\n" + "=" * 70)
    print("TESTING MODES")
    print("=" * 70)
    print("1. Interactive Mode - Test your own examples")
    print("2. Batch Test Mode - Test predefined examples")
    print("3. Quick Test - Single example")
    print("4. Exit")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == '1':
        interactive_mode(model)
    elif choice == '2':
        batch_test_mode(model)
    elif choice == '3':
        print("\n" + "=" * 70)
        print("QUICK TEST")
        print("=" * 70)
        print_detailed_prediction(model, 'Max', 7, 3, 9)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
