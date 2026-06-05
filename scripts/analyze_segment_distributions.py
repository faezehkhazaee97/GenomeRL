"""Analyze and compare segment length distributions: RL vs stride-pooling vs Gumbel."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from collections import Counter

from src.data.dataset import GenomicsDataset
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.stride_pooling_classifier import StridedPoolingClassifier
from src.models.gumbel_segmenter_env import GumbelSegmenterEnvironment


def get_segment_lengths(model, task, model_type, device, n_samples=200):
    """Extract segment lengths from a trained model on test set."""

    dataset = GenomicsDataset(data_dir="./data", task=task)
    segment_lengths = []

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            seq, label = dataset[i]

            if model_type == "rl":
                logits, boundaries, compression = model(seq, device)
                # Count segments (transitions from 0 to 1 in boundaries)
                boundaries_np = boundaries.cpu().numpy()
                num_segments = np.sum(np.diff(boundaries_np) > 0) + 1
                segment_lengths.append(num_segments)

            elif model_type == "stride":
                stride = model.stride
                # For stride pooling, segments are deterministic
                num_segments = len(seq) // stride
                segment_lengths.append(max(1, num_segments))

            elif model_type == "gumbel":
                logits, boundaries, compression = model(seq, device)
                boundaries_np = boundaries.cpu().numpy()
                num_segments = np.sum(np.diff(boundaries_np) > 0) + 1
                segment_lengths.append(num_segments)

    return segment_lengths


def main():
    """Analyze segment distributions for promoter and splice-site."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = ["promoter", "splice_site"]
    output_dir = Path("results/segment_distributions")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for task in tasks:
        print(f"\nAnalyzing {task}...")

        # Load models
        rl_model = FineTunedDNABERT2Environment(
            backbone_name="zhihan1996/DNABERT-2-117M",
            checkpoint_path="./pretrained_models/DNABERT2_117M.pt",
        ).to(device)
        rl_model.load_state_dict(
            torch.load(f"results/grpo_finetuned/{task}/best_model.pt", map_location=device)
        )
        rl_model.eval()

        stride_model = StridedPoolingClassifier(
            backbone_name="zhihan1996/DNABERT-2-117M",
            checkpoint_path="./pretrained_models/DNABERT2_117M.pt",
            num_labels=2,
            stride=4,
        ).to(device)
        stride_model.eval()

        # Get segment lengths
        rl_lengths = get_segment_lengths(rl_model, task, "rl", device)
        stride_lengths = get_segment_lengths(stride_model, task, "stride", device)

        # Compute statistics
        def compute_stats(lengths, method):
            return {
                "Task": task,
                "Method": method,
                "Mean": np.mean(lengths),
                "Median": np.median(lengths),
                "Std": np.std(lengths),
                "Min": np.min(lengths),
                "Max": np.max(lengths),
                "P25": np.percentile(lengths, 25),
                "P75": np.percentile(lengths, 75),
            }

        all_stats.append(compute_stats(rl_lengths, "RL-learned"))
        all_stats.append(compute_stats(stride_lengths, "Fixed-stride (s=4)"))

        # Create distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(rl_lengths, bins=20, alpha=0.7, label="RL-learned", edgecolor="black")
        axes[0].axvline(np.mean(rl_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(rl_lengths):.1f}")
        axes[0].set_xlabel("Number of Segments")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"RL Segment Distribution ({task})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(stride_lengths, bins=20, alpha=0.7, label="Fixed-stride (s=4)", color="orange", edgecolor="black")
        axes[1].axvline(np.mean(stride_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(stride_lengths):.1f}")
        axes[1].set_xlabel("Number of Segments")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Stride-pooling Distribution ({task})")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"segment_distribution_{task}.png", dpi=150)
        print(f"  Saved: segment_distribution_{task}.png")
        plt.close()

    # Summary table
    df = pd.DataFrame(all_stats)
    print("\n=== Segment Length Statistics ===\n")
    print(df.to_string(index=False))

    df.to_csv(output_dir / "segment_length_statistics.csv", index=False)
    print(f"\nSaved to {output_dir / 'segment_length_statistics.csv'}")

    # Key insights
    print("\n=== Key Insights ===")
    for task in tasks:
        rl_stats = df[(df["Task"] == task) & (df["Method"] == "RL-learned")].iloc[0]
        stride_stats = df[(df["Task"] == task) & (df["Method"] == "Fixed-stride (s=4)")].iloc[0]

        rl_mean = rl_stats["Mean"]
        stride_mean = stride_stats["Mean"]

        print(f"\n{task}:")
        print(f"  RL avg segments: {rl_mean:.1f} (std: {rl_stats['Std']:.1f})")
        print(f"  Stride avg segments: {stride_mean:.1f} (std: {stride_stats['Std']:.1f})")

        if abs(rl_mean - stride_mean) / stride_mean < 0.1:
            print(f"  → Similar compression strategies")
        elif rl_mean > stride_mean:
            print(f"  → RL learns finer granularity (+{(rl_mean/stride_mean - 1)*100:.1f}%)")
        else:
            print(f"  → RL learns coarser segments ({(1 - rl_mean/stride_mean)*100:.1f}% fewer)")


if __name__ == "__main__":
    main()
