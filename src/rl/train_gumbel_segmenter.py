"""Train the Gumbel-Softmax differentiable segmentation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.metrics import compute_classification_metrics
from src.models.gumbel_segmenter import GumbelSegmenter
from src.utils.config import load_config
from src.utils.seed import set_seed


def evaluate(model: GumbelSegmenter, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    comp_ratios = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            logits, stats = model(sequences, device, hard=True)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            n += len(labels)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            comp_ratios.append(stats["compression_ratio"])

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(n, 1)
    metrics["avg_compression_ratio"] = sum(comp_ratios) / max(len(comp_ratios), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config["env"]
    gumbel_cfg = config.get("gumbel", {})

    model = GumbelSegmenter(
        backbone_name=env_cfg["backbone_name"],
        checkpoint_path=env_cfg["checkpoint_path"],
        hidden_size=env_cfg["hidden_size"],
        num_labels=env_cfg["num_labels"],
        trainable_encoder_layers=gumbel_cfg.get("trainable_encoder_layers", 2),
        initial_temperature=gumbel_cfg.get("initial_temperature", 1.0),
        min_temperature=gumbel_cfg.get("min_temperature", 0.1),
        anneal_rate=gumbel_cfg.get("anneal_rate", 0.0003),
        beta=gumbel_cfg.get("beta", 0.02),
        max_length=env_cfg.get("max_length", 512),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    bs = config["training"].get("batch_size", 16)
    train_loader = DataLoader(GUEPromoterDataset(config["dataset"]["train_path"]),
                              batch_size=bs, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(GUEPromoterDataset(config["dataset"]["valid_path"]),
                              batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(GUEPromoterDataset(config["dataset"]["test_path"]),
                              batch_size=bs, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"].get("learning_rate", 1e-5)),
        weight_decay=0.01,
    )

    num_epochs = config["training"].get("num_epochs", 5)
    beta = gumbel_cfg.get("beta", 0.02)
    best_f1, best_state = 0.0, None
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0.0, 0

        for batch in train_loader:
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            logits, stats = model(sequences, device)
            cls_loss = F.cross_entropy(logits, labels)
            comp_loss = beta * stats["compression_ratio"]
            loss = cls_loss + comp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.anneal_temperature()

            epoch_loss += loss.item() * len(labels)
            epoch_n += len(labels)

        vm = evaluate(model, valid_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"loss={epoch_loss/epoch_n:.4f} | "
            f"valid_acc={vm['accuracy']:.4f} | "
            f"valid_f1={vm['macro_f1']:.4f} | "
            f"comp={vm['avg_compression_ratio']:.4f} | "
            f"temp={model.temperature:.4f}"
        )
        history.append({"epoch": epoch, **vm, "temperature": model.temperature})

        if vm["macro_f1"] >= best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
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
