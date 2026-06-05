"""Evaluation metrics for GenomeRL baselines."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    labels: List[int],
    predictions: List[int],
    probabilities: Optional[List[List[float]]] = None,
) -> Dict[str, float]:
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    result = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    if probabilities is not None:
        probs = np.array(probabilities)
        classes = sorted(set(labels))
        try:
            if len(classes) == 2:
                # Binary: use probability of positive class
                auroc = roc_auc_score(labels, probs[:, 1])
            else:
                # Multi-class: macro one-vs-rest AUROC
                lab_bin = label_binarize(labels, classes=classes)
                auroc = roc_auc_score(lab_bin, probs, average="macro", multi_class="ovr")
            result["auroc"] = float(auroc)
        except Exception:
            pass  # AUROC undefined for degenerate predictions
    return result
