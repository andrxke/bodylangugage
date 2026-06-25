"""
ST-GCN training script.

Trains a lightweight ST-GCN model for binary gesture classification
on labeled skeleton clips. Supports class weighting for imbalanced
data, cosine learning rate scheduling, and early stopping.

Usage:
    # Train a "facing" classifier
    python -m training.train --data-dir data/ --gesture facing

    # Custom hyperparameters
    python -m training.train --data-dir data/ --gesture facing \\
        --epochs 100 --lr 0.001 --batch-size 8 --clip-length 128

    # Resume from checkpoint
    python -m training.train --data-dir data/ --gesture facing --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from classification.gesture_labels import GESTURE_TYPES
from classification.graph import MediaPipeGraph
from classification.stgcn import STGCN
from training.dataset import GestureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        # Add person dimension: (N, C, T, V) → (N, C, T, V, 1)
        x = x.to(device)
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping for training stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Returns:
        Dict with loss, accuracy, precision, recall, f1, and
        confusion matrix counts (tp, fp, tn, fn).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    if total == 0:
        return {
            "loss": 0.0, "accuracy": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }

    # Confusion matrix for the positive class (label=1).
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def print_confusion_matrix(metrics: dict[str, float]) -> None:
    """Pretty-print the confusion matrix."""
    tp, fp = metrics["tp"], metrics["fp"]
    tn, fn = metrics["tn"], metrics["fn"]

    print("\n  Confusion Matrix:")
    print("                Predicted")
    print("                Neg   Pos")
    print(f"  Actual Neg  [{tn:4d}  {fp:4d} ]")
    print(f"         Pos  [{fn:4d}  {tp:4d} ]")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an ST-GCN gesture classifier.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory (with clips/ and labels.csv).",
    )
    parser.add_argument(
        "--gesture",
        required=True,
        choices=list(GESTURE_TYPES.keys()),
        help="Gesture type to train.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/gesture_classifiers",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs. Default: 100.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size. Default: 8.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate. Default: 0.001.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=128,
        help="Fixed temporal length for clips. Default: 128.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs). Default: 15.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability. Default: 0.3.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation. Default: 0.2.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    device = get_device()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gesture_type = args.gesture
    gesture_info = GESTURE_TYPES[gesture_type]

    logger.info(
        "Training gesture classifier: %s (%s)",
        gesture_type,
        gesture_info["description"],
    )

    # -----------------------------------------------------------------------
    # Load dataset.
    # -----------------------------------------------------------------------

    full_dataset = GestureDataset(
        data_dir=str(data_dir),
        gesture_type=gesture_type,
        clip_length=args.clip_length,
        augment=False,  # Will create separate augmented dataset for train.
    )

    if len(full_dataset) == 0:
        logger.error("Dataset is empty. Cannot train.")
        sys.exit(1)

    if len(full_dataset) < 4:
        logger.error(
            "Dataset has only %d samples. Need at least 4 for "
            "train/val split.",
            len(full_dataset),
        )
        sys.exit(1)

    # Split into train / val.
    train_indices, val_indices = full_dataset.get_split_indices(
        val_fraction=args.val_fraction,
    )

    # Create augmented training dataset.
    train_dataset = GestureDataset(
        data_dir=str(data_dir),
        gesture_type=gesture_type,
        clip_length=args.clip_length,
        augment=True,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)  # No augmentation for val.

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------------------------
    # Build model.
    # -----------------------------------------------------------------------

    graph = MediaPipeGraph()
    model = STGCN(
        num_classes=gesture_info["num_classes"],
        graph=graph,
        in_channels=3,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Count parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (%.1fK)", num_params, num_params / 1000)

    # Class weighting for imbalanced data.
    class_weights = full_dataset.get_class_weights().to(device)
    logger.info("Class weights: %s", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Resume from checkpoint if requested.
    start_epoch = 0
    best_val_f1 = -1.0
    best_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }

    model_path = output_dir / f"{gesture_type}.pt"
    checkpoint_path = output_dir / f"{gesture_type}_checkpoint.pt"

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        logger.info(
            "Resumed from epoch %d (best F1: %.3f)",
            start_epoch,
            best_val_f1,
        )

    # -----------------------------------------------------------------------
    # Training loop.
    # -----------------------------------------------------------------------

    epochs_without_improvement = 0
    train_start = time.time()

    logger.info("Starting training for %d epochs...", args.epochs)
    print(
        f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>6} | {'LR':>8}"
    )
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs):
        # Train.
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Evaluate.
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch + 1:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
            f"{val_loss:>8.4f} | {val_acc:>6.1%} | {val_f1:>5.3f} | "
            f"{current_lr:>8.6f}"
        )

        # Save best model.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0

            # Save model weights.
            torch.save(model.state_dict(), model_path)

            # Save checkpoint (for resuming).
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                },
                checkpoint_path,
            )

            logger.debug("  → New best model saved (F1: %.3f)", val_f1)
        else:
            epochs_without_improvement += 1

        # Early stopping.
        if epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1,
                args.patience,
            )
            break

        scheduler.step()

    train_elapsed = time.time() - train_start

    # -----------------------------------------------------------------------
    # Final report.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"Training complete: {epoch + 1} epochs in {train_elapsed:.1f}s")
    print(f"Best model saved: {model_path}")
    print(f"  Val Accuracy:  {best_metrics['accuracy']:.1%}")
    print(f"  Val Precision: {best_metrics['precision']:.1%}")
    print(f"  Val Recall:    {best_metrics['recall']:.1%}")
    print(f"  Val F1:        {best_metrics['f1']:.3f}")
    print_confusion_matrix(best_metrics)

    # Save config alongside the model.
    config = {
        "gesture_type": gesture_type,
        "num_classes": gesture_info["num_classes"],
        "in_channels": 3,
        "clip_length": args.clip_length,
        "graph": "MediaPipeGraph",
        "num_nodes": graph.num_nodes,
        "dropout": args.dropout,
        "training": {
            "epochs_trained": epoch + 1,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "training_time_seconds": round(train_elapsed, 1),
        },
        "best_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in best_metrics.items()
        },
    }

    config_path = output_dir / f"{gesture_type}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Config saved: %s", config_path)

    # Clean up checkpoint.
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
