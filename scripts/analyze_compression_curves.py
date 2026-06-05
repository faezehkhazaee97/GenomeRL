"""
Analyze compression vs. performance curves across all methods and tasks.
Generates comparison plots for paper revision.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results/baselines")
RL_RUNS_DIR = Path("results/rl_runs")
TASKS = [
    "promoter",
    "human_tf_0",
    "human_tf_1",
    "human_tf_2",
    "human_tf_3",
    "human_tf_4",
    "splice_site",
]

def load_metrics(path):
    """Load metrics from JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        m = json.load(f)
        # Normalize field names
        if "macro_f1" in m and "f1" not in m:
            m["f1"] = m["macro_f1"]
        if "avg_compression_ratio" in m and "compression_ratio" not in m:
            m["compression_ratio"] = m["avg_compression_ratio"]
        if "avg_segments" in m and "compression_ratio" not in m:
            m["compression_ratio"] = m["avg_segments"]
        return m

def collect_all_results():
    """Collect results from all methods and tasks."""
    data = defaultdict(list)

    # Baseline (fine-tuned DNABERT-2)
    for task in TASKS:
        baseline_path = RESULTS_DIR / f"dnabert2_bpe_finetuned" / "metrics.json"
        if task != "promoter":
            baseline_path = RESULTS_DIR / f"dnabert2_bpe_finetuned_{task}" / "metrics.json"

        metrics = load_metrics(baseline_path)
        if metrics:
            data["Baseline"].append({
                "task": task,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "compression": metrics.get("compression_ratio", 1.0),
                "avg_tokens": metrics.get("avg_tokens"),
            })

    # Stride-pooling
    for task in TASKS:
        sp_path = RESULTS_DIR / f"stride_pooling_{task}" / "metrics.json"
        metrics = load_metrics(sp_path)
        if metrics:
            data["Stride-pooling"].append({
                "task": task,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "compression": metrics.get("compression_ratio"),
                "avg_tokens": metrics.get("avg_tokens"),
            })

    # Gumbel-Softmax
    for task in TASKS:
        gumbel_path = RESULTS_DIR / f"gumbel_segmenter_{task}" / "metrics.json"
        metrics = load_metrics(gumbel_path)
        if metrics:
            data["Gumbel-Softmax"].append({
                "task": task,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "compression": metrics.get("compression_ratio"),
                "avg_tokens": metrics.get("avg_tokens"),
            })

    # RL (fine-tuned, sampled mode)
    for task in TASKS:
        rl_path = RL_RUNS_DIR / f"grpo_finetuned_env_{task}" / "eval_test_sampled.json"
        if task == "promoter":
            rl_path = RL_RUNS_DIR / "grpo_finetuned_env" / "eval_test_sampled.json"

        metrics = load_metrics(rl_path)
        if metrics:
            data["RL (FT)"].append({
                "task": task,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "compression": metrics.get("compression_ratio"),
                "avg_tokens": metrics.get("avg_tokens"),
            })

    # Lagrangian RL (if available)
    lag_path = RL_RUNS_DIR / "grpo_finetuned_env_lagrangian" / "eval_test_sampled.json"
    if lag_path.exists():
        metrics = load_metrics(lag_path)
        if metrics:
            data["Lagrangian RL"].append({
                "task": "promoter",
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "compression": metrics.get("compression_ratio"),
                "avg_tokens": metrics.get("avg_tokens"),
            })

    return data

def plot_compression_curves(data, task):
    """Plot compression vs. F1 for a single task."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "Baseline": "black",
        "Stride-pooling": "blue",
        "Gumbel-Softmax": "orange",
        "RL (FT)": "red",
        "Lagrangian RL": "purple",
    }

    # F1 vs compression
    for method, results in data.items():
        task_results = [r for r in results if r["task"] == task]
        if task_results:
            r = task_results[0]
            if r.get("compression") and r.get("f1"):
                ax1.scatter(r["compression"], r["f1"], s=100, label=method, color=colors.get(method, "gray"))
                ax1.annotate(method, (r["compression"], r["f1"]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax1.set_xlabel("Compression Ratio", fontsize=11)
    ax1.set_ylabel("F1 Score", fontsize=11)
    ax1.set_title(f"Compression vs F1: {task}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)

    # Accuracy vs compression
    for method, results in data.items():
        task_results = [r for r in results if r["task"] == task]
        if task_results:
            r = task_results[0]
            if r.get("compression") and r.get("accuracy"):
                ax2.scatter(r["compression"], r["accuracy"], s=100, label=method, color=colors.get(method, "gray"))

    ax2.set_xlabel("Compression Ratio", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"Compression vs Accuracy: {task}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig

def print_comparison_table(data):
    """Print comprehensive comparison table."""
    print("\n" + "="*120)
    print("COMPRESSION-PERFORMANCE COMPARISON ACROSS ALL METHODS AND TASKS")
    print("="*120 + "\n")

    for task in TASKS:
        print(f"\n{task.upper()}")
        print("-" * 120)
        print(f"{'Method':<20} {'F1':<10} {'Accuracy':<12} {'Compression':<15} {'Avg Tokens':<15}")
        print("-" * 120)

        for method in ["Baseline", "Stride-pooling", "Gumbel-Softmax", "RL (FT)", "Lagrangian RL"]:
            if method in data:
                task_results = [r for r in data[method] if r["task"] == task]
                if task_results:
                    r = task_results[0]
                    f1 = r.get("f1") or 0
                    acc = r.get("accuracy") or 0
                    comp = r.get("compression") or 0
                    tokens = r.get("avg_tokens") or 0
                    if f1 is not None and acc is not None:
                        print(f"{method:<20} {f1:<10.4f} {acc:<12.4f} {comp:<15.4f} {tokens:<15.2f}")

if __name__ == "__main__":
    print("Collecting results...")
    data = collect_all_results()

    print_comparison_table(data)

    print("\n\nGenerating plots...")
    for task in TASKS:
        fig = plot_compression_curves(data, task)
        output_path = f"results/analysis/compression_curve_{task}.pdf"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close(fig)

    print("\nDone!")
