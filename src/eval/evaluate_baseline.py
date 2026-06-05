"""Train and evaluate the first supervised baseline classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.metrics import compute_classification_metrics
from src.models.backbone import FrozenBackboneClassifier
from src.models.simple_dna_cnn import SimpleDNACNN, encode_dna_batch
from src.utils.config import load_config
from src.utils.seed import set_seed


def tokenize_batch(tokenizer, sequences, max_length: int, device: str):
    encoded = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def count_bpe_tokens(tokenizer, sequences):
    counts = []
    for sequence in sequences:
        counts.append(len(tokenizer.tokenize(sequence)))
    return counts


def maybe_limit_batches(dataloader, max_batches: int | None):
    if max_batches is None:
        return dataloader
    return islice(dataloader, max_batches)


def evaluate_transformer(
    model,
    tokenizer,
    dataloader,
    max_length: int,
    device: str,
    max_batches: int | None = None,
):
    model.eval()
    all_labels = []
    all_predictions = []
    all_token_counts = []
    all_sequence_lengths = []

    with torch.no_grad():
        for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Evaluating"):
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            encoded = tokenize_batch(tokenizer, sequences, max_length=max_length, device=device)
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            predictions = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_token_counts.extend(count_bpe_tokens(tokenizer, sequences))
            all_sequence_lengths.extend([len(sequence) for sequence in sequences])

    metrics = compute_classification_metrics(all_labels, all_predictions)
    avg_tokens = sum(all_token_counts) / len(all_token_counts)
    avg_seq_len = sum(all_sequence_lengths) / len(all_sequence_lengths)
    metrics.update(
        {
            "avg_tokens": float(avg_tokens),
            "avg_sequence_length": float(avg_seq_len),
            "compression_ratio": float(avg_tokens / avg_seq_len),
        }
    )
    return metrics


def train_one_epoch_transformer(
    model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    max_length: int,
    device: str,
    max_batches: int | None = None,
):
    model.train()
    total_loss = 0.0
    num_steps = 0
    for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Training"):
        sequences = batch["sequences"]
        labels = batch["labels"].to(device)
        encoded = tokenize_batch(tokenizer, sequences, max_length=max_length, device=device)
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_steps += 1
    return total_loss / max(num_steps, 1)


def evaluate_cnn(model, dataloader, device: str, max_batches: int | None = None):
    model.eval()
    all_labels = []
    all_predictions = []
    all_sequence_lengths = []

    with torch.no_grad():
        for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Evaluating"):
            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            input_ids = encode_dna_batch(sequences).to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_sequence_lengths.extend([len(sequence) for sequence in sequences])

    metrics = compute_classification_metrics(all_labels, all_predictions)
    avg_seq_len = sum(all_sequence_lengths) / len(all_sequence_lengths)
    metrics.update(
        {
            "avg_tokens": float(avg_seq_len),
            "avg_sequence_length": float(avg_seq_len),
            "compression_ratio": 1.0,
        }
    )
    return metrics


def train_one_epoch_cnn(model, dataloader, optimizer, criterion, device: str, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    num_steps = 0
    for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Training"):
        sequences = batch["sequences"]
        labels = batch["labels"].to(device)
        input_ids = encode_dna_batch(sequences).to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_steps += 1
    return total_loss / max(num_steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dnabert2", "cnn"],
        default="dnabert2",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device = config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Loading dataset files...")

    train_dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    valid_dataset = GUEPromoterDataset(config["dataset"]["valid_path"])
    test_dataset = GUEPromoterDataset(config["dataset"]["test_path"])
    print(
        f"Dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}"
    )

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_epochs = args.epochs or config["training"]["epochs"]
    output_dir = Path(config["evaluation"]["output_dir"])
    if args.model_type == "cnn":
        output_dir = output_dir.parent / "cnn"
    elif args.model_type == "dnabert2":
        output_dir = output_dir.parent / "dnabert2_bpe"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"
    best_valid_f1 = -1.0
    criterion = nn.CrossEntropyLoss()

    if args.model_type == "dnabert2":
        backbone_name = config["model"]["backbone_name"]
        print(f"Loading tokenizer: {backbone_name}")
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
        print("Tokenizer loaded.")
        print(f"Loading model: {backbone_name}")
        model = FrozenBackboneClassifier(
            backbone_name=backbone_name,
            hidden_size=config["model"]["hidden_size"],
            num_labels=config["model"]["num_labels"],
            freeze_backbone=config["model"]["freeze_backbone"],
        ).to(device)
        print("Model loaded and moved to device.")
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        print(f"Trainable parameter tensors: {len(trainable_params)}")
        optimizer = torch.optim.AdamW(
            trainable_params,
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
    else:
        model = SimpleDNACNN(num_labels=config["model"]["num_labels"]).to(device)
        print("CNN model loaded and moved to device.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        for epoch in range(1, num_epochs + 1):
            print("=" * 80)
            print(f"Epoch {epoch}/{num_epochs}")
            print("=" * 80)
            train_loss = train_one_epoch_cnn(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_batches=args.max_train_batches,
            )
            print(f"Train loss: {train_loss:.4f}")
            valid_metrics = evaluate_cnn(
                model=model,
                dataloader=valid_loader,
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
        test_metrics = evaluate_cnn(
            model=model,
            dataloader=test_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )

    print("=" * 80)
    print("Final test evaluation")
    print("=" * 80)
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
