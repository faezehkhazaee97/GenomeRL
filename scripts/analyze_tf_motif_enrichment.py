"""Analyze boundary enrichment at known TF motifs across transfer tasks.

Compares boundary probability distributions at known JASPAR motif positions
vs. off-motif positions for each TF-binding task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# JASPAR motif consensus sequences (common TFs in human_tf tasks)
MOTIFS = {
    "TATA": "TATAAA",  # TATA-box (also in TF tasks as baseline)
    "CAAT": "GGCCAATCT",  # CAAT-box
    "GC": "GGGCGG",  # GC-box
    "SP1": "GGGCGG",  # SP1 (same as GC)
    "E_box": "CACGTG",  # E-box
    "NFY": "CCAAT",  # NFY/CAAT
}


def find_motif_positions(sequence: str, motif: str, tolerance: int = 1) -> list[int]:
    """Find approximate motif matches with up to `tolerance` mismatches."""
    positions = []
    seq = sequence.upper()
    mot = motif.upper()
    motlen = len(mot)

    for i in range(len(seq) - motlen + 1):
        window = seq[i : i + motlen]
        mismatches = sum(a != b for a, b in zip(window, mot))
        if mismatches <= tolerance:
            positions.append(i)
    return positions


def analyze_task(task: str, seed: int = 42) -> dict:
    """Analyze boundary enrichment for a single TF task."""
    # Load boundary probabilities
    run_dir = Path(
        f"results/rl_runs/grpo_finetuned_env_{task}_seed_{seed}"
        if seed != 42
        else f"results/rl_runs/grpo_finetuned_env_{task}"
    )
    eval_file = run_dir / "eval_test_sampled.json"

    if not eval_file.exists():
        return {"task": task, "error": "no eval file"}

    # Load test sequences
    data_file = Path(f"data/processed/gue_{task}/test.jsonl")
    if not data_file.exists():
        return {"task": task, "error": "no data file"}

    sequences = []
    with open(data_file) as f:
        for line in f:
            sequences.append(json.loads(line)["sequence"].upper())

    # For now, report that we have the data structure ready
    result = {
        "task": task,
        "num_sequences": len(sequences),
        "avg_seq_len": np.mean([len(s) for s in sequences]),
        "status": "motif analysis pending (requires boundary probability checkpoint)",
    }

    return result


if __name__ == "__main__":
    tasks = ["human_tf_0", "human_tf_1", "human_tf_2", "human_tf_3", "human_tf_4"]

    print("TF-binding task motif enrichment analysis")
    print("=" * 60)

    for task in tasks:
        result = analyze_task(task)
        print(f"\n{task}:")
        for k, v in result.items():
            if k != "task":
                print(f"  {k}: {v}")

    print("\nNote: Full motif enrichment analysis requires:")
    print("1. Boundary probability outputs from RL policy")
    print("2. Per-task JASPAR motif database")
    print("3. Sequence alignment against JASPAR")
    print("\nThis can be run once TF-specific trained models provide probabilities.")
