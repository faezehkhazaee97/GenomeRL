"""Analyze threshold robustness: confusion matrices and class-wise metrics at different thresholds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.data.dataset import GenomicsDataset
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment


def analyze_threshold_robustness(task, model_path, thresholds=[0.05, 0.11, 0.30, 0.50]):
    """Analyze how threshold affects classification metrics at different compression levels."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FineTunedDNABERT2Environment(
        backbone_name="zhihan1996/DNABERT-2-117M",
        checkpoint_path="./pretrained_models/DNABERT2_117M.pt",
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test data
    dataset = GenomicsDataset(data_dir="./data", task=task)

    results = {
        "task": task,
        "thresholds": [],
    }

    with torch.no_grad():
        for threshold in thresholds:
            print(f"  Analyzing threshold τ={threshold}")

            all_preds = []
            all_labels = []
            all_compressions = []

            for i in range(min(100, len(dataset))):  # Subsample for speed
                seq, label = dataset[i]
                sequences = [seq]
                labels = torch.tensor([label])

                logits, boundaries, compression = model(sequences, device, threshold=threshold)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().item())
                all_labels.append(label)
                all_compressions.append(compression)

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            avg_compression = np.mean(all_compressions)

            # Compute metrics
            acc = (all_preds == all_labels).mean()
            f1 = f1_score(all_labels, all_preds, average="binary" if len(np.unique(all_labels)) == 2 else "weighted", zero_division=0)
            precision = precision_score(all_labels, all_preds, average="binary" if len(np.unique(all_labels)) == 2 else "weighted", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="binary" if len(np.unique(all_labels)) == 2 else "weighted", zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            results["thresholds"].append(
                {
                    "threshold": threshold,
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "avg_compression": float(avg_compression),
                    "confusion_matrix": cm.tolist(),
                    "n_samples": len(all_labels),
                }
            )

    return results


def main():
    """Analyze threshold robustness for promoter and splice-site tasks."""

    tasks = ["promoter", "splice_site"]
    results_dir = Path("results/threshold_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for task in tasks:
        print(f"Analyzing {task}...")
        model_path = f"results/grpo_finetuned/{task}/best_model.pt"

        if not Path(model_path).exists():
            print(f"  Model not found: {model_path}")
            continue

        task_results = analyze_threshold_robustness(task, model_path)
        all_results.append(task_results)

        # Save per-task
        with open(results_dir / f"{task}_threshold_analysis.json", "w") as f:
            json.dump(task_results, f, indent=2)

    # Aggregate to CSV
    rows = []
    for task_result in all_results:
        task = task_result["task"]
        for thresh in task_result["thresholds"]:
            rows.append(
                {
                    "Task": task,
                    "Threshold": thresh["threshold"],
                    "Accuracy": thresh["accuracy"],
                    "F1": thresh["f1"],
                    "Precision": thresh["precision"],
                    "Recall": thresh["recall"],
                    "Compression": thresh["avg_compression"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "threshold_robustness.csv", index=False)

    print("\n=== Threshold Robustness Summary ===\n")
    print(df.to_string(index=False))
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
