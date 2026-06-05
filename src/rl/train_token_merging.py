"""Train token merging baseline classifier on GUE tasks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.dataset import GenomicsDataset
from src.models.token_merging_env import TokenMergingDNABERT2Environment
from src.utils.seed import set_seed


def train_epoch(model, train_loader, optimizer, device, num_labels):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    for sequences, labels in tqdm(train_loader, desc="Training"):
        labels = labels.to(device)
        logits, _ = model(sequences, device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(train_loader)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc = (all_logits.argmax(dim=1) == all_labels).float().mean().item()

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, val_loader, device, num_labels):
    """Evaluate on validation set."""
    model.eval()
    all_logits = []
    all_labels = []
    compression_ratios = []

    for sequences, labels in tqdm(val_loader, desc="Evaluating"):
        labels = labels.to(device)
        logits, stats = model(sequences, device)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        if "compression_ratio" in stats:
            compression_ratios.append(stats["compression_ratio"])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = all_logits.argmax(dim=1)

    acc = accuracy_score(all_labels.numpy(), preds.numpy())
    f1 = f1_score(all_labels.numpy(), preds.numpy(), average="binary" if num_labels == 2 else "weighted")

    avg_compression = np.mean(compression_ratios) if compression_ratios else 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "compression_ratio": avg_compression,
    }


def main():
    parser = argparse.ArgumentParser(description="Train token merging baseline")
    parser.add_argument("--task", type=str, required=True, help="Task name (promoter, human_tf_0, etc.)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--backbone_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--checkpoint_path", type=str, default="./pretrained_models/DNABERT2_117M.pt")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--merge_ratio", type=float, default=0.5, help="Fraction of tokens to keep after merging")
    parser.add_argument("--merge_start_layer", type=int, default=6, help="Layer to start merging from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./results/token_merging")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = GenomicsDataset(
        data_dir=args.data_dir,
        task=args.task,
    )
    num_labels = len(dataset.label_map)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = TokenMergingDNABERT2Environment(
        backbone_name=args.backbone_name,
        checkpoint_path=args.checkpoint_path,
        num_labels=num_labels,
        merge_ratio=args.merge_ratio,
        merge_start_layer=args.merge_start_layer,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    results = {
        "config": vars(args),
        "epochs": [],
    }

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, num_labels)
        val_metrics = evaluate(model, val_loader, device, num_labels)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, Compression: {val_metrics['compression_ratio']:.4f}")

        results["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            }
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save final results
    results["best_epoch"] = best_epoch
    results["best_f1"] = best_f1
    results["best_metrics"] = results["epochs"][best_epoch]

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
