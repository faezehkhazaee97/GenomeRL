"""Run boundary interpretability analysis for an expanded set of promoter motifs.

Runs analyze_boundary_interpretability for each motif in a predefined list and
saves results to results/analysis/. Uses the finetuned-env agent checkpoint.

Motifs analysed:
  - TATA  (TATA box — canonical core promoter element)
  - CAAT  (CAAT box)
  - GCGC  (GC-rich element)
  - GGGCGG (SP1 binding site)
  - CACGTG (E-box / MYC binding site)
  - CCGCCC (GC-box / SP1 variant)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = "results/rl_runs/grpo_seed_42/agent_classifier_last.pt"
CONFIG = "configs/grpo.yaml"
THRESHOLD = "0.11"
SPLIT = "test"

MOTIFS = [
    ("TATA",   "results/analysis/boundary_interpretability_expanded_tata.json"),
    ("CAAT",   "results/analysis/boundary_interpretability_expanded_caat.json"),
    ("GCGC",   "results/analysis/boundary_interpretability_expanded_gcgc.json"),
    ("GGGCGG", "results/analysis/boundary_interpretability_expanded_sp1.json"),
    ("CACGTG", "results/analysis/boundary_interpretability_expanded_ebox.json"),
    ("CCGCCC", "results/analysis/boundary_interpretability_expanded_gcbox.json"),
]


def run_motif(motif: str, output_path: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Motif: {motif}  →  {output_path}")
    cmd = [
        sys.executable, "-m", "src.eval.analyze_boundary_interpretability",
        "--config", CONFIG,
        "--checkpoint", CHECKPOINT,
        "--motif", motif,
        "--threshold", THRESHOLD,
        "--split", SPLIT,
        "--output-path", output_path,
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"WARNING: motif {motif} exited with code {result.returncode}")


def main() -> None:
    Path(REPO_ROOT / "results" / "analysis").mkdir(parents=True, exist_ok=True)

    summary = {}
    for motif, output_path in MOTIFS:
        run_motif(motif, output_path)
        full_path = REPO_ROOT / output_path
        if full_path.exists():
            with full_path.open() as fh:
                data = json.load(fh)
            summary[motif] = {
                "avg_boundary_rate_on_motif": data.get("avg_boundary_rate_on_motif"),
                "avg_boundary_rate_off_motif": data.get("avg_boundary_rate_off_motif"),
                "motif_sequence_fraction": data.get("motif_sequence_fraction"),
                "threshold": data.get("threshold"),
            }

    summary_path = REPO_ROOT / "results" / "analysis" / "expanded_motif_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nExpanded motif summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
