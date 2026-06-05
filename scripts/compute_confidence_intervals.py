"""Compute confidence intervals on main task comparisons using multi-seed results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_ci(values, confidence=0.95):
    """Compute confidence interval for a set of values."""
    if len(values) < 2:
        return (values[0] if values else 0, values[0] if values else 0)

    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return (mean - ci, mean + ci)


def main():
    """Extract multi-seed F1 results and compute CIs."""

    results_dir = Path("results")
    output_file = Path("results/confidence_intervals.csv")

    all_results = []

    # Collect results from multi-seed experiments
    for task_dir in sorted(results_dir.glob("grpo_finetuned/*/")):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        results_file = task_dir / "results.json"

        if results_file.exists():
            try:
                with open(results_file) as f:
                    task_results = json.load(f)

                # Extract best metrics
                best_metrics = task_results.get("best_metrics", {})
                f1 = best_metrics.get("f1", 0)
                acc = best_metrics.get("accuracy", 0)

                all_results.append(
                    {
                        "Task": task_name,
                        "Method": "Standard RL",
                        "F1": f1,
                        "Accuracy": acc,
                    }
                )
            except Exception as e:
                print(f"Error loading {task_name}: {e}")

    # Also collect Lagrangian results
    for task_dir in sorted(results_dir.glob("grpo_finetuned_lagrangian/*/")):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        results_file = task_dir / "results.json"

        if results_file.exists():
            try:
                with open(results_file) as f:
                    task_results = json.load(f)

                best_metrics = task_results.get("best_metrics", {})
                f1 = best_metrics.get("f1", 0)
                acc = best_metrics.get("accuracy", 0)

                all_results.append(
                    {
                        "Task": task_name,
                        "Method": "Lagrangian RL",
                        "F1": f1,
                        "Accuracy": acc,
                    }
                )
            except Exception as e:
                print(f"Error loading Lagrangian {task_name}: {e}")

    if not all_results:
        print("No results found")
        return

    df = pd.DataFrame(all_results)

    # Compute CIs by task and method
    ci_results = []

    for task in df["Task"].unique():
        for method in df["Method"].unique():
            task_method_data = df[(df["Task"] == task) & (df["Method"] == method)]

            if len(task_method_data) > 0:
                f1_values = task_method_data["F1"].values
                acc_values = task_method_data["Accuracy"].values

                f1_mean = np.mean(f1_values)
                f1_ci = compute_ci(f1_values)
                acc_mean = np.mean(acc_values)
                acc_ci = compute_ci(acc_values)

                ci_results.append(
                    {
                        "Task": task,
                        "Method": method,
                        "F1 Mean": f1_mean,
                        "F1 95% CI Lower": f1_ci[0],
                        "F1 95% CI Upper": f1_ci[1],
                        "Accuracy Mean": acc_mean,
                        "Accuracy 95% CI Lower": acc_ci[0],
                        "Accuracy 95% CI Upper": acc_ci[1],
                        "N Seeds": len(f1_values),
                    }
                )

    ci_df = pd.DataFrame(ci_results)

    print("\n=== Confidence Intervals (95%) on Main Comparisons ===\n")
    print(ci_df.to_string(index=False))

    ci_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Highlight comparisons where CIs overlap vs don't overlap
    print("\n=== Statistical Significance ===")
    for task in ci_df["Task"].unique():
        std_row = ci_df[(ci_df["Task"] == task) & (ci_df["Method"] == "Standard RL")]
        lagr_row = ci_df[(ci_df["Task"] == task) & (ci_df["Method"] == "Lagrangian RL")]

        if len(std_row) > 0 and len(lagr_row) > 0:
            std_f1 = std_row.iloc[0]["F1 Mean"]
            lagr_f1 = lagr_row.iloc[0]["F1 Mean"]
            std_ci = (std_row.iloc[0]["F1 95% CI Lower"], std_row.iloc[0]["F1 95% CI Upper"])
            lagr_ci = (lagr_row.iloc[0]["F1 95% CI Lower"], lagr_row.iloc[0]["F1 95% CI Upper"])

            # Check if CIs overlap
            if lagr_ci[0] > std_ci[1] or std_ci[0] > lagr_ci[1]:
                direction = "Lagrangian better" if lagr_f1 > std_f1 else "Standard RL better"
                print(f"✓ {task}: {direction} (no CI overlap)")
            else:
                print(f"~ {task}: No significant difference (CIs overlap)")


if __name__ == "__main__":
    main()
