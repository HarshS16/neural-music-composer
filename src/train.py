"""
Step 4: Training Loop
======================
Trains the MusicLSTM on processed MIDI sequences.

Usage:
    python src/train.py --midi_dir data/midi --epochs 50 --device cuda

What to expect:
    Epoch 5:   loss ~4.0  (random noise)
    Epoch 15:  loss ~2.5  (scale-like patterns)
    Epoch 30:  loss ~1.8  (recognizable phrases)
    Epoch 50:  loss ~1.3  (musical structure)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parse_midi import process_midi_directory, save_processed_data
from src.dataset import create_dataloader
from src.model import MusicLSTM, count_parameters


def train(model, dataloader, epochs, lr, device, save_dir="outputs"):
    """
    Full training loop with logging, checkpointing, and loss curve.
    """
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    loss_history = []
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training on {device} | LR={lr} | Epochs={epochs}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()

        progress = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{epochs}",
                        leave=False, ncols=80)

        for batch_x, batch_y in progress:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            logits, _ = model(batch_x)

            # Loss: compare predicted distribution vs actual next token
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # (batch*seq, vocab)
                batch_y.view(-1)                    # (batch*seq,)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — CRITICAL for RNNs
            # Without this, gradients can explode and training diverges
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        elapsed = time.time() - start_time

        # Print every 5 epochs (or first/last)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, os.path.join(save_dir, "best_model.pt"))

    # Save final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
    }, os.path.join(save_dir, "final_model.pt"))

    # Save loss curve
    _plot_loss_curve(loss_history, save_dir)

    # Save training config
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump({
            "epochs": epochs,
            "lr": lr,
            "final_loss": avg_loss,
            "best_loss": best_loss,
            "loss_history": loss_history
        }, f, indent=2)

    print(f"\n✓ Training complete! Best loss: {best_loss:.4f}")
    print(f"  Model saved to {save_dir}/best_model.pt")
    print(f"  Loss curve saved to {save_dir}/loss_curve.png")

    return loss_history


def _plot_loss_curve(loss_history, save_dir):
    """Save a clean loss curve plot (great for the article)."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history,
             color="#FF6B6B", linewidth=2)
    plt.fill_between(range(1, len(loss_history) + 1), loss_history,
                     alpha=0.1, color="#FF6B6B")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Training Loss — LSTM Music Generator", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()


# ---------- CLI Entry Point ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM Music Generator")
    parser.add_argument("--midi_dir", type=str, default="data/midi")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'mps', 'cpu', or 'auto'")
    parser.add_argument("--save_dir", type=str, default="outputs")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    # Step 1: Process MIDI files
    sequences, token_to_idx, idx_to_token, stats = process_midi_directory(
        args.midi_dir, fs=8
    )
    save_processed_data(sequences, token_to_idx)

    # Step 2: Create DataLoader
    dataloader, vocab_size = create_dataloader(
        sequences, token_to_idx,
        seq_length=args.seq_length,
        batch_size=args.batch_size
    )

    # Step 3: Create model
    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    count_parameters(model)

    # Step 4: Train
    loss_history = train(
        model, dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )

    # Save vocab alongside model for generation
    with open(os.path.join(args.save_dir, "vocab.json"), "w") as f:
        json.dump({
            "token_to_idx": {str(k): v for k, v in token_to_idx.items()},
            "idx_to_token": {str(v): k for k, v in token_to_idx.items()},
            "vocab_size": vocab_size,
            "stats": stats
        }, f, indent=2)

    print(f"\n🎵 Ready to generate! Run:")
    print(f"   python src/generate.py --model {args.save_dir}/best_model.pt")
