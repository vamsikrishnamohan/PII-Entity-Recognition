import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    ap.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    return ap.parse_args()


def compute_class_weights(dataset, num_labels, smoothing=0.3):
    """Compute class weights to handle imbalanced data with smoothing."""
    label_counts = np.zeros(num_labels)
    
    for item in dataset:
        for label in item['labels']:
            if label != -100:  # Ignore padding
                label_counts[label] += 1
    
    # Inverse frequency weighting with smoothing to prevent extreme weights
    total = label_counts.sum()
    weights = total / (num_labels * label_counts + 1e-6)
    
    # Apply smoothing: blend with uniform weights
    # smoothing=0 -> full inverse freq, smoothing=1 -> all weights=1
    uniform_weight = 1.0
    weights = (1 - smoothing) * weights + smoothing * uniform_weight
    
    # Normalize
    weights = weights / weights.mean()
    
    return torch.FloatTensor(weights)


def evaluate_model(model, dev_dl, device):
    """Evaluate model on dev set and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_dl, desc="Evaluating", leave=False):
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    model.train()
    return avg_loss


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    print(f"Loading training data from {args.train}")
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    print(f"Loading dev data from {args.dev}")
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")
    
    # Optionally compute class weights from training data
    class_weights = None
    if args.use_class_weights:
        print("\nComputing class weights for imbalanced data...")
        class_weights = compute_class_weights(train_ds, len(LABELS))
        
        print("Class weights:")
        for i, (label, weight) in enumerate(zip(LABELS, class_weights)):
            print(f"  {label:15s}: {weight:.3f}")
    else:
        print("\nClass weights: DISABLED (training without weights)")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    if args.use_class_weights:
        print(f"\nCreating model from {args.model_name} with class weights")
    else:
        print(f"\nCreating model from {args.model_name} (no class weights)")
    model = create_model(args.model_name, class_weights=class_weights)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Early stopping variables
    best_dev_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(args.out_dir, "best_model")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Device: {args.device}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient clipping: {args.grad_clip}")
    print(f"Early stopping patience: {args.patience}\n")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_dl))
        
        # Validation phase
        avg_dev_loss = evaluate_model(model, dev_dl, args.device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")
        
        # Early stopping and best model saving
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  → New best model saved (dev loss: {best_dev_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement (patience: {patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Save final model (last epoch)
    final_model_path = os.path.join(args.out_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n✓ Training complete!")
    print(f"  - Best model saved to: {best_model_path} (dev loss: {best_dev_loss:.4f})")
    print(f"  - Final model saved to: {final_model_path}")
    print(f"\nFor inference, use: {best_model_path}")


if __name__ == "__main__":
    main()
