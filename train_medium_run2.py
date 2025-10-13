import sys
sys.path.insert(0, 'src')

import numpy as np
import random

# DIFFERENT SEED for second run
random.seed(123)  # Changed from 42
np.random.seed(123)  # Changed from 42

from vocabluary import VOCAB_SIZE
from data_generatpr import generate_tokenized_dataset, split_dataset
from data_utils import create_batch
from transformer import Transformer
from loss import CrossEntropyLoss
from optimizer import Adam
from train_utils import train_step, evaluate
from evaluation import MetricsTracker, ModelCheckpoint, print_operation_analysis

print("=" * 70)
print("SECOND TRAINING RUN - Different Random Seed (123)")
print("=" * 70)

# Generate dataset with NEW random seed
print("\nGenerating dataset...")
dataset = generate_tokenized_dataset(
    num_examples=10000,
    num_args=3,
    max_value=9,
    balance_operations=True
)

train_data, val_data, test_data = split_dataset(dataset)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Prepare batches
train_inputs, train_targets, _ = create_batch(train_data, max_length=20)
train_targets = train_targets[:, 0]

val_inputs, val_targets, _ = create_batch(val_data, max_length=20)
val_targets = val_targets[:, 0]

test_inputs, test_targets, _ = create_batch(test_data, max_length=20)
test_targets = test_targets[:, 0]

# Create model
print("\nCreating model...")
model = Transformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    ffn_dim=256,
    max_seq_len=50
)

# Training setup
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.get_parameters, learning_rate=0.001)
tracker = MetricsTracker()
checkpoint = ModelCheckpoint('checkpoints/')

# Training loop
print("\nStarting training...")
num_epochs = 20
batch_size = 32

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(train_inputs))
    train_inputs_shuffled = train_inputs[shuffle_idx]
    train_targets_shuffled = train_targets[shuffle_idx]
    
    # Train
    num_batches = (len(train_inputs) + batch_size - 1) // batch_size
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
    
    train_loss = epoch_loss / len(train_inputs)
    train_acc = epoch_acc / len(train_inputs)
    
    # Validate
    val_loss, val_acc = evaluate(model, val_inputs, val_targets, loss_fn, batch_size=batch_size)
    
    tracker.update(train_loss, train_acc, val_loss, val_acc)
    
    print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")
    
    if tracker.is_best_epoch():
        print(f"  🌟 New best!")
        checkpoint.save(model, optimizer, tracker, epoch, filename='best_model_run2.pkl')

# Test evaluation
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

test_loss, test_acc = evaluate(model, test_inputs, test_targets, loss_fn, batch_size=batch_size)
print(f"\nTest Accuracy: {test_acc:.4f}")

print("\nPer-Operation Analysis:")
print_operation_analysis(model, test_data, ['First', 'Second', 'Last', 'Max', 'Min'])

