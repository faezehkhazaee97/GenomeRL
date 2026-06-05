"""Train and evaluate a fine-tuned DNABERT-2 promoter baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.evaluate_baseline import (
    evaluate_transformer,
    train_one_epoch_transformer,
)
from src.models.backbone import FrozenBackboneClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_finetuned_dnabert2.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    config["seed"] = seed
    set_seed(seed)

    device = config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    train_dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    valid_dataset = GUEPromoterDataset(config["dataset"]["valid_path"])
    test_dataset = GUEPromoterDataset(config["dataset"]["test_path"])

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_epochs = args.epochs or config["training"]["epochs"]
    output_dir = Path(args.output_dir or config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"
    best_valid_f1 = -1.0
    criterion = nn.CrossEntropyLoss()

    backbone_name = config["model"]["backbone_name"]
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
    model = FrozenBackboneClassifier(
        backbone_name=backbone_name,
        hidden_size=config["model"]["hidden_size"],
        num_labels=config["model"]["num_labels"],
        freeze_backbone=config["model"].get("freeze_backbone", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    for epoch in range(1, num_epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{num_epochs}")
        print("=" * 80)
        train_loss = train_one_epoch_transformer(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            max_length=config["tokenizer"]["max_length"],
            device=device,
            max_batches=args.max_train_batches,
        )
        print(f"Train loss: {train_loss:.4f}")
        valid_metrics = evaluate_transformer(
            model=model,
            tokenizer=tokenizer,
            dataloader=valid_loader,
            max_length=config["tokenizer"]["max_length"],
            device=device,
            max_batches=args.max_eval_batches,
        )
        print("Validation metrics:")
        print(json.dumps(valid_metrics, indent=2))
        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_transformer(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        max_length=config["tokenizer"]["max_length"],
        device=device,
        max_batches=args.max_eval_batches,
    )

    print("=" * 80)
    print("Final test evaluation")
    print(json.dumps(test_metrics, indent=2))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
