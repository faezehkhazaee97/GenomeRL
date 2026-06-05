"""Per-class precision/recall, ECE, and sequence-length stratification for splice-site.

Addresses reviewer Q1/Q2: provides confusion matrix, per-class F1, calibration
metrics, and length-stratified analysis to characterize the RL failure mode.

Splice-site is a 3-class task: 0=no-splice, 1=acceptor, 2=donor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.models.backbone import FrozenBackboneClassifier
from src.models.token_gating_agent import sample_boundary_mask
from src.rl.rollout import group_token_states
from transformers import AutoTokenizer
from src.utils.seed import set_seed


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(labels))


def evaluate_model(model, dataloader, device, mode="baseline"):
    """Run inference and collect per-sample predictions, probabilities, labels, lengths."""
    if mode == "baseline":
        model.eval()
    else:
        env, agent = model
        env.eval()
        agent.eval()
    all_labels, all_preds, all_probs, all_lengths = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch["sequences"]
            labs = batch["labels"].to(device)

            if mode == "baseline":
                tokenizer = model._tokenizer
                encoded = tokenizer(
                    seqs, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(device)
                attn_mask = encoded["attention_mask"].to(device)
                logits = model(input_ids, attn_mask)
            else:
                feats = env.encode_sequences(seqs, device)
                _, bprobs = agent(feats["token_states"], feats["token_mask"])
                masks, _ = sample_boundary_mask(bprobs, feats["token_mask"])
                grouped = group_token_states(feats["token_states"], masks, feats["token_mask"])
                logits = env.logits_from_segments(
                    feats["cls_states"], grouped["segment_states"], grouped["segment_mask"]
                )

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labs.cpu().tolist())
            all_lengths.extend([len(s) for s in seqs])

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_lengths),
    )


def analyze_and_report(labels, preds, probs, lengths, name: str) -> dict:
    """Compute full diagnostics for one model."""
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    ece = compute_ece(probs, labels)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    report = classification_report(
        labels, preds, labels=[0, 1, 2],
        target_names=["no-splice", "acceptor", "donor"],
        zero_division=0, output_dict=True,
    )

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(f"ECE:       {ece:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"           no-splice  acceptor  donor")
    for i, row_name in enumerate(["no-splice", "acceptor ", "donor    "]):
        print(f"  {row_name}: {cm[i]}")
    print(f"\nPer-class report:")
    for cls in ["no-splice", "acceptor", "donor"]:
        r = report[cls]
        print(f"  {cls:12s}: P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}  n={int(r['support'])}")

    # Length-stratified analysis
    short = lengths <= 300
    medium = (lengths > 300) & (lengths <= 400)
    long_ = lengths > 400
    print(f"\nLength-stratified macro-F1:")
    for mask, name_ in [(short, "≤300bp"), (medium, "301-400bp"), (long_, ">400bp")]:
        if mask.sum() > 0:
            f = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
            print(f"  {name_}: F1={f:.4f}  (n={mask.sum()})")

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            cls: {k: round(v, 4) for k, v in report[cls].items() if k != "support"}
            for cls in ["no-splice", "acceptor", "donor"]
        },
        "length_stratified": {
            "le_300": round(float(f1_score(labels[short], preds[short], average="macro", zero_division=0)), 4) if short.sum() > 0 else None,
            "301_400": round(float(f1_score(labels[medium], preds[medium], average="macro", zero_division=0)), 4) if medium.sum() > 0 else None,
            "gt_400": round(float(f1_score(labels[long_], preds[long_], average="macro", zero_division=0)), 4) if long_.sum() > 0 else None,
        },
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = GUEPromoterDataset("data/processed/gue_splice_site/test.jsonl")
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    results = []

    # 1. Fine-tuned DNABERT-2 baseline
    print("\nLoading fine-tuned DNABERT-2 baseline...")
    baseline = FrozenBackboneClassifier(
        backbone_name="zhihan1996/DNABERT-2-117M",
        hidden_size=768,
        num_labels=3,
        freeze_backbone=False,
    ).to(device)
    ckpt = torch.load("results/baselines/dnabert2_bpe_finetuned_splice_site/best_model.pt", map_location="cpu")
    baseline.load_state_dict(ckpt, strict=False)
    baseline._tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    labels, preds, probs, lengths = evaluate_model(baseline, loader, device, mode="baseline")
    results.append(analyze_and_report(labels, preds, probs, lengths, "Fine-tuned DNABERT-2 baseline"))

    # 2. Fine-tuned RL (standard)
    print("\nLoading fine-tuned RL splice-site checkpoint...")
    env = FineTunedDNABERT2Environment(
        backbone_name="zhihan1996/DNABERT-2-117M",
        checkpoint_path="results/baselines/dnabert2_bpe/best_model.pt",
        hidden_size=768, num_labels=3, max_length=512,
        cls_mix_weight=0.0, trainable_encoder_layers=2, freeze_embeddings=True,
    ).to(device)
    agent = TokenStateBoundaryAgent(input_dim=768, model_dim=256, hidden_dim=128).to(device)

    rl_ckpt = torch.load("results/rl_runs/grpo_finetuned_env_splice_site/agent_env_last.pt", map_location="cpu")
    env.load_state_dict(rl_ckpt["env_state_dict"], strict=False)
    agent.load_state_dict(rl_ckpt["agent_state_dict"], strict=False)
    labels, preds, probs, lengths = evaluate_model((env, agent), loader, device, mode="rl")
    results.append(analyze_and_report(labels, preds, probs, lengths, "Fine-tuned RL (standard)"))

    # 3. Stride pooling baseline (uses fine-tuned baseline model with fixed stride)
    print("\nStride pooling: using baseline model predictions as proxy (full tokens)")
    # Note: stride pooling uses the FT baseline architecture; report same metrics as baseline
    # for comparison with the RL failure mode

    output = Path("results/analysis/splice_site_perclass_analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
