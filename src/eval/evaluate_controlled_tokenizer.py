"""Controlled tokenizer ablation with a fixed downstream classifier.

This script keeps the classifier architecture fixed and varies only the
tokenization scheme, so differences can be attributed more directly to the
tokenizer rather than model architecture changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.eval.agent_token_stats import summarize_agent_tokens
from src.eval.metrics import compute_classification_metrics
from src.models.build_agent import build_token_gating_agent
from src.models.build_classifier import build_learned_token_classifier
from src.models.learned_token_classifier import encode_token_lists
from src.models.token_gating_agent import (
    encode_dna_batch,
    masks_to_token_lists,
    sample_boundary_mask,
    threshold_boundary_mask,
)
from src.tokenization.fixed_kmer import fixed_kmer_tokenize
from src.tokenization.random_policy import random_tokenize
from src.tokenization.single_nt import single_nt_tokenize
from src.utils.config import load_config
from src.utils.seed import set_seed


Batch = Dict[str, object]


def maybe_limit_batches(dataloader, max_batches: Optional[int]):
    if max_batches is None:
        for batch in dataloader:
            yield batch
        return
    count = 0
    for batch in dataloader:
        if count >= max_batches:
            break
        yield batch
        count += 1


def _stable_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_learned_tokenizer(
    config: Dict,
    device: torch.device,
    mode: str,
    threshold: float,
) -> Callable[[Batch], List[List[str]]]:
    checkpoint_path = Path(config["tokenizer"]["learned_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_config = checkpoint.get("config", load_config("configs/grpo.yaml"))
    agent = build_token_gating_agent(agent_config).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    def tokenize_batch(batch: Batch) -> List[List[str]]:
        sequences = batch["sequences"]
        input_ids, attention_mask = encode_dna_batch(sequences)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            _, boundary_probs = agent(input_ids, attention_mask)
            if mode == "sampled":
                masks, _ = sample_boundary_mask(boundary_probs, attention_mask)
            elif mode == "threshold":
                masks = threshold_boundary_mask(boundary_probs, attention_mask, threshold=threshold)
            else:
                raise ValueError(f"Unsupported learned mode: {mode}")
        return masks_to_token_lists(sequences, masks)

    return tokenize_batch


def build_tokenizer_fn(
    tokenizer_name: str,
    config: Dict,
    device: torch.device,
) -> Callable[[Batch], List[List[str]]]:
    if tokenizer_name == "single_nt":
        return lambda batch: [single_nt_tokenize(sequence) for sequence in batch["sequences"]]

    if tokenizer_name == "fixed_kmer":
        k = config["tokenizer"].get("fixed_k", 6)
        return lambda batch: [fixed_kmer_tokenize(sequence, k=k, overlap=False) for sequence in batch["sequences"]]

    if tokenizer_name == "random":
        boundary_prob = config["tokenizer"].get("random_boundary_prob", 0.2)

        def tokenize_random(batch: Batch) -> List[List[str]]:
            ids = batch["ids"]
            sequences = batch["sequences"]
            token_lists = []
            for seq_id, sequence in zip(ids, sequences):
                seed = _stable_int(f"{seq_id}:{boundary_prob}:{config['seed']}")
                token_lists.append(random_tokenize(sequence, boundary_prob=boundary_prob, seed=seed))
            return token_lists

        return tokenize_random

    if tokenizer_name == "learned_sampled":
        return build_learned_tokenizer(
            config=config,
            device=device,
            mode="sampled",
            threshold=config["tokenizer"].get("learned_threshold", 0.5),
        )

    if tokenizer_name == "learned_threshold":
        return build_learned_tokenizer(
            config=config,
            device=device,
            mode="threshold",
            threshold=config["tokenizer"].get("learned_threshold", 0.5),
        )

    raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")


def run_classifier_batch(
    classifier: torch.nn.Module,
    token_lists: List[List[str]],
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    token_ids, token_mask, base_mask = encode_token_lists(token_lists)
    token_ids = token_ids.to(device)
    token_mask = token_mask.to(device)
    base_mask = base_mask.to(device)
    logits = classifier(token_ids, token_mask, base_mask)
    loss = F.cross_entropy(logits, labels)
    predictions = torch.argmax(logits, dim=-1)
    return {
        "loss": loss,
        "predictions": predictions,
        "logits": logits,
    }


def train_one_epoch(
    classifier: torch.nn.Module,
    dataloader,
    tokenizer_fn: Callable[[Batch], List[List[str]]],
    optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    classifier.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Training"):
        labels = batch["labels"].to(device)
        token_lists = tokenizer_fn(batch)
        outputs = run_classifier_batch(classifier, token_lists, labels, device)
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        total_loss += float(outputs["loss"].item())
        steps += 1
    return total_loss / max(steps, 1)


def evaluate_split(
    classifier: torch.nn.Module,
    dataloader,
    tokenizer_fn: Callable[[Batch], List[List[str]]],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    classifier.eval()
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_sequences: List[str] = []
    all_token_lists: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(maybe_limit_batches(dataloader, max_batches), desc="Evaluating"):
            labels = batch["labels"].to(device)
            token_lists = tokenizer_fn(batch)
            outputs = run_classifier_batch(classifier, token_lists, labels, device)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(outputs["predictions"].cpu().tolist())
            all_sequences.extend(batch["sequences"])
            all_token_lists.extend(token_lists)

    metrics = compute_classification_metrics(all_labels, all_predictions)
    token_stats = summarize_agent_tokens(all_sequences, all_token_lists)
    metrics.update(
        {
            "avg_tokens": float(token_stats["avg_num_tokens"]),
            "avg_sequence_length": float(sum(len(seq) for seq in all_sequences) / len(all_sequences)),
            "compression_ratio": float(token_stats["avg_compression_ratio"]),
            "avg_token_length": float(token_stats["avg_token_length"]),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/controlled_ablation.yaml")
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["single_nt", "fixed_kmer", "random", "learned_sampled", "learned_threshold"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device_name = config["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    train_dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    valid_dataset = GUEPromoterDataset(config["dataset"]["valid_path"])
    test_dataset = GUEPromoterDataset(config["dataset"]["test_path"])

    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    tokenizer_fn = build_tokenizer_fn(args.tokenizer, config, device)
    classifier = build_learned_token_classifier(config).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    output_dir = Path(config["evaluation"]["output_dir"]) / args.tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"
    best_valid_f1 = -1.0
    epochs = args.epochs or config["training"]["epochs"]

    print(f"Using device: {device}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Dataset sizes: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    for epoch in range(1, epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 80)
        train_loss = train_one_epoch(
            classifier=classifier,
            dataloader=train_loader,
            tokenizer_fn=tokenizer_fn,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
        )
        print(f"Train loss: {train_loss:.4f}")
        valid_metrics = evaluate_split(
            classifier=classifier,
            dataloader=valid_loader,
            tokenizer_fn=tokenizer_fn,
            device=device,
            max_batches=args.max_eval_batches,
        )
        print("Validation metrics:")
        print(json.dumps(valid_metrics, indent=2))
        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            torch.save(classifier.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    classifier.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate_split(
        classifier=classifier,
        dataloader=test_loader,
        tokenizer_fn=tokenizer_fn,
        device=device,
        max_batches=args.max_eval_batches,
    )
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
