"""Analyze Lagrangian RL compression variance and tightness of constraint enforcement."""

import json
from pathlib import Path

import pandas as pd


def main():
    """Extract compression statistics from Lagrangian RL logs."""

    lagrangian_dir = Path("results/grpo_finetuned_lagrangian")
    target_compression = 0.18

    results = []

    print("=== Lagrangian RL Compression Analysis ===\n")

    for task_dir in sorted(lagrangian_dir.glob("*")):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        results_file = task_dir / "results.json"

        if not results_file.exists():
            print(f"Skipping {task_name} (no results.json)")
            continue

        try:
            with open(results_file) as f:
                task_results = json.load(f)

            # Extract compression info
            best_metrics = task_results.get("best_metrics", {})
            compression = best_metrics.get("compression_ratio", 0)
            f1 = best_metrics.get("f1", 0)
            accuracy = best_metrics.get("accuracy", 0)

            # Calculate variance from target
            compression_variance = compression - target_compression
            compression_variance_pct = (compression_variance / target_compression) * 100

            results.append(
                {
                    "Task": task_name,
                    "Target Compression": target_compression,
                    "Achieved Compression": compression,
                    "Variance from Target": compression_variance,
                    "Variance %": compression_variance_pct,
                    "F1": f1,
                    "Accuracy": accuracy,
                }
            )

        except Exception as e:
            print(f"Error loading {task_name}: {e}")

    if results:
        df = pd.DataFrame(results)

        # Calculate aggregate stats
        mean_compression = df["Achieved Compression"].mean()
        mean_variance = df["Variance from Target"].mean()
        mean_variance_pct = df["Variance %"].mean()

        print(df.to_string(index=False))
        print("\n=== Summary Statistics ===")
        print(f"Average achieved compression: {mean_compression:.4f}")
        print(f"Target compression: {target_compression:.4f}")
        print(f"Average variance from target: {mean_variance:.4f} ({mean_variance_pct:.1f}%)")
        print(f"Compression std dev: {df['Achieved Compression'].std():.4f}")

        # Save to CSV
        output_file = Path("results/lagrangian_compression_analysis.csv")
        df.to_csv(output_file, index=False)
        print(f"\nAnalysis saved to {output_file}")

        # Interpretation
        print("\n=== Interpretation ===")
        if mean_variance < 0:
            print(f"✓ Lagrangian achieved TIGHTER compression (avg {mean_variance*100:.1f}% below target)")
        else:
            print(f"⚠ Lagrangian over-shot compression (avg {mean_variance*100:.1f}% above target)")

        print(f"Tightness of constraint: {'High' if df['Variance %'].abs().mean() < 10 else 'Moderate' if df['Variance %'].abs().mean() < 20 else 'Loose'}")


if __name__ == "__main__":
    main()
