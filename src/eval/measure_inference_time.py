"""Measure inference throughput: plain DNABERT-2 vs RL-grouped vs random-grouped.

Outputs a JSON file with ms/sequence, sequences/second, and token counts for
each method so the speedup can be reported in the paper.

Usage:
    python -m src.eval.measure_inference_time \
        --config configs/grpo_finetuned_env.yaml \
        --checkpoint results/rl_runs/grpo_finetuned_env/agent_env_last.pt \
        --output-path results/analysis/inference_timing.json \
        --n-warmup 10 --n-batches 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_gating_agent import sample_boundary_mask
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.rl.rollout import group_token_states
from src.utils.config import load_config


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_batches(fn, dataloader, n_warmup: int, n_batches: int, device: torch.device):
    """Run fn(batch) for n_warmup + n_batches batches, return timing stats."""
    times: List[float] = []
    seq_counts: List[int] = []

    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i >= n_warmup + n_batches - 1:
            break

    # warmup (GPU JIT, cache priming)
    for batch in batches[:n_warmup]:
        fn(batch)
        sync(device)

    # timed runs
    for batch in batches[n_warmup:]:
        sync(device)
        t0 = time.perf_counter()
        fn(batch)
        sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)          # ms
        seq_counts.append(len(batch["sequences"]))

    total_seq = sum(seq_counts)
    total_ms = sum(times)
    return {
        "ms_per_batch": total_ms / len(times),
        "ms_per_sequence": total_ms / total_seq,
        "sequences_per_second": total_seq / (total_ms / 1000),
        "n_batches": len(times),
        "n_sequences": total_seq,
    }


# ──────────────────────────────────────────────
# Inference functions
# ──────────────────────────────────────────────

def make_plain_dnabert2_fn(env, device: torch.device):
    """Plain DNABERT-2 inference: encode → pool CLS → classify (no RL grouping).

    Uses the env's already-loaded backbone to avoid a conflicting second load.
    This is equivalent to the fine-tuned DNABERT-2 baseline at inference time.
    """
    env.eval()
    token_counts: List[float] = []

    @torch.no_grad()
    def _run(batch):
        seqs = batch["sequences"]
        features = env.encode_sequences(seqs, device)
        # Use CLS token directly (no grouping) — this is the plain baseline path
        cls_states = features["cls_states"]          # [B, hidden]
        token_mask = features["token_mask"]
        n_tokens = token_mask.sum(dim=1).float().mean().item()
        token_counts.append(n_tokens)
        env.classifier(cls_states)                   # classify from CLS only

    _run.token_counts = token_counts
    return _run


def make_rl_grouped_fn(agent, env, device: torch.device):
    """RL-grouped inference: encode → agent boundary → group → classify."""
    agent.eval()
    env.eval()
    token_counts: List[float] = []

    @torch.no_grad()
    def _run(batch):
        seqs = batch["sequences"]
        features = env.encode_sequences(seqs, device)
        token_states = features["token_states"]
        token_mask = features["token_mask"]

        _, boundary_probs = agent(token_states, token_mask)
        masks, _ = sample_boundary_mask(boundary_probs, token_mask)

        grouped = group_token_states(token_states, masks, token_mask)
        grouped_states = grouped["segment_states"]
        grouped_mask = grouped["segment_mask"]
        n_groups = grouped_mask.sum(dim=1).float().mean().item()
        token_counts.append(n_groups)
        env.logits_from_segments(features["cls_states"], grouped_states, grouped_mask)

    _run.token_counts = token_counts
    return _run


def make_random_grouped_fn(env, device: torch.device, boundary_prob: float = 0.174):
    """Random grouping at matched compression (~11/63 ≈ 0.174 per position)."""
    env.eval()
    token_counts: List[float] = []

    @torch.no_grad()
    def _run(batch):
        seqs = batch["sequences"]
        features = env.encode_sequences(seqs, device)
        token_states = features["token_states"]
        token_mask = features["token_mask"]

        B, L, _ = token_states.shape
        rand = torch.rand(B, L, device=device)
        masks = (rand < boundary_prob).long() * token_mask
        masks[:, 0] = 1  # always start a segment at position 0

        grouped = group_token_states(token_states, masks, token_mask)
        grouped_states = grouped["segment_states"]
        grouped_mask = grouped["segment_mask"]
        n_groups = grouped_mask.sum(dim=1).float().mean().item()
        token_counts.append(n_groups)
        env.logits_from_segments(features["cls_states"], grouped_states, grouped_mask)

    _run.token_counts = token_counts
    return _run


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-path", default="results/analysis/inference_timing.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="Warmup batches (excluded from timing)")
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Timed batches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    split_path = cfg["dataset"][f"{args.split}_path"]
    dataset = GUEPromoterDataset(split_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    env_cfg = cfg["env"]
    backbone_name = env_cfg["backbone_name"]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Build env
    env = FineTunedDNABERT2Environment(
        backbone_name=env_cfg["backbone_name"],
        checkpoint_path=env_cfg["checkpoint_path"],
        hidden_size=env_cfg["hidden_size"],
        num_labels=env_cfg["num_labels"],
        max_length=env_cfg.get("max_length", 512),
        cls_mix_weight=env_cfg.get("cls_mix_weight", 0.0),
        trainable_encoder_layers=env_cfg.get("trainable_encoder_layers", 4),
        freeze_embeddings=env_cfg.get("freeze_embeddings", True),
    ).to(device)
    if "env_state_dict" in ckpt:
        env.load_state_dict(ckpt["env_state_dict"])

    # Build agent
    agent_cfg = cfg["agent"]
    agent = TokenStateBoundaryAgent(
        input_dim=agent_cfg["input_dim"],
        model_dim=agent_cfg.get("model_dim", 256),
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        num_layers=agent_cfg.get("num_layers", 1),
        dropout=agent_cfg.get("dropout", 0.1),
        initial_boundary_prob=agent_cfg.get("initial_boundary_prob", 0.2),
    ).to(device)
    agent_state = ckpt.get("agent_state_dict", ckpt.get("agent", ckpt))
    agent.load_state_dict(agent_state, strict=False)

    print("Timing plain DNABERT-2...")
    plain_fn = make_plain_dnabert2_fn(env, device)
    plain_stats = time_batches(plain_fn, dataloader, args.n_warmup, args.n_batches, device)
    plain_stats["avg_tokens"] = float(
        sum(plain_fn.token_counts) / max(len(plain_fn.token_counts), 1))

    print("Timing RL-grouped inference...")
    rl_fn = make_rl_grouped_fn(agent, env, device)
    rl_stats = time_batches(rl_fn, dataloader, args.n_warmup, args.n_batches, device)
    rl_stats["avg_tokens"] = float(
        sum(rl_fn.token_counts) / max(len(rl_fn.token_counts), 1))

    print("Timing random-grouped inference (matched compression)...")
    rnd_fn = make_random_grouped_fn(env, device)
    rnd_stats = time_batches(rnd_fn, dataloader, args.n_warmup, args.n_batches, device)
    rnd_stats["avg_tokens"] = float(
        sum(rnd_fn.token_counts) / max(len(rnd_fn.token_counts), 1))

    # Speedup relative to plain DNABERT-2
    baseline_ms = plain_stats["ms_per_sequence"]
    rl_stats["speedup_vs_plain"] = baseline_ms / rl_stats["ms_per_sequence"]
    rnd_stats["speedup_vs_plain"] = baseline_ms / rnd_stats["ms_per_sequence"]

    results = {
        "device": str(device),
        "batch_size": args.batch_size,
        "n_warmup_batches": args.n_warmup,
        "n_timed_batches": args.n_batches,
        "plain_dnabert2": plain_stats,
        "rl_grouped": rl_stats,
        "random_grouped_matched": rnd_stats,
    }

    print("\n=== Results ===")
    for method, stats in [("Plain DNABERT-2", plain_stats),
                           ("RL-grouped", rl_stats),
                           ("Random-grouped", rnd_stats)]:
        print(f"{method:30s}  {stats['ms_per_sequence']:.2f} ms/seq  "
              f"{stats['sequences_per_second']:.0f} seq/s  "
              f"avg_tokens={stats['avg_tokens']:.1f}  "
              f"speedup={stats.get('speedup_vs_plain', 1.0):.2f}x")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
