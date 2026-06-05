"""Aggregate token merging baseline results and compare to RL."""

import json
from pathlib import Path

import pandas as pd


def main():
    """Compile token merging results into comparison tables."""

    results_dir = Path("results/token_merging")
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Collect results
    all_results = []

    for task_dir in sorted(results_dir.iterdir()):
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

            # Get best epoch metrics
            best_metrics = task_results["best_metrics"]
            config = task_results["config"]

            all_results.append(
                {
                    "Task": task_name,
                    "MergeRatio": config.get("merge_ratio", "N/A"),
                    "Accuracy": best_metrics.get("accuracy", 0),
                    "F1": best_metrics.get("f1", 0),
                    "Compression": best_metrics.get("compression_ratio", 0),
                    "BestEpoch": task_results.get("best_epoch", -1),
                }
            )
        except Exception as e:
            print(f"Error loading {task_name}: {e}")

    if not all_results:
        print("No results found")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Sort by task and compression
    df = df.sort_values(["Task", "Compression"])

    print("\n=== Token Merging Baseline Results ===\n")
    print(df.to_string(index=False))

    # Save to CSV
    output_file = Path("results/token_merging_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===\n")
    print(f"Average F1 across tasks: {df['F1'].mean():.4f}")
    print(f"Average Compression: {df['Compression'].mean():.4f}")
    print(f"F1 std dev: {df['F1'].std():.4f}")


if __name__ == "__main__":
    main()
