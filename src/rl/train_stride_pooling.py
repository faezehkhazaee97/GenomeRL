"""Train a fixed-stride mean-pooling classifier as a non-RL baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.metrics import compute_classification_metrics
from src.models.stride_pooling_classifier import StridedPoolingClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


def evaluate(model: StridedPoolingClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    comp_ratios = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            logits, stats = model(sequences, device)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            n += len(labels)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            comp_ratios.append(stats["compression_ratio"])

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(n, 1)
    metrics["avg_segments"] = sum(comp_ratios) / len(comp_ratios) if comp_ratios else 0.0
    metrics["avg_compression_ratio"] = sum(comp_ratios) / len(comp_ratios) if comp_ratios else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config["env"]
    model_cfg = config.get("stride_pooling", {})

    model = StridedPoolingClassifier(
        backbone_name=env_cfg["backbone_name"],
        checkpoint_path=env_cfg["checkpoint_path"],
        hidden_size=env_cfg["hidden_size"],
        num_labels=env_cfg["num_labels"],
        stride=model_cfg.get("stride", 5),
        trainable_encoder_layers=model_cfg.get("trainable_encoder_layers", 2),
        max_length=env_cfg.get("max_length", 512),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    train_ds = GUEPromoterDataset(config["dataset"]["train_path"])
    valid_ds = GUEPromoterDataset(config["dataset"]["valid_path"])
    test_ds  = GUEPromoterDataset(config["dataset"]["test_path"])

    bs = config["training"].get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"].get("learning_rate", 1e-5)),
        weight_decay=0.01,
    )

    num_epochs = config["training"].get("num_epochs", 5)
    best_valid_f1, best_state = 0.0, None
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0.0, 0

        for batch in train_loader:
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits, _ = model(sequences, device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            epoch_n += len(labels)

        valid_metrics = evaluate(model, valid_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={epoch_loss/epoch_n:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f} | "
            f"valid_f1={valid_metrics['macro_f1']:.4f} | "
            f"comp={valid_metrics['avg_compression_ratio']:.4f}"
        )
        history.append({"epoch": epoch, **valid_metrics})

        if valid_metrics["macro_f1"] >= best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: acc={test_metrics['accuracy']:.4f}  f1={test_metrics['macro_f1']:.4f}  "
          f"comp={test_metrics['avg_compression_ratio']:.4f}")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
