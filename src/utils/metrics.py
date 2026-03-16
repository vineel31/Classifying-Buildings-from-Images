"""Metrics computation for classification."""
import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    labels: List[int],
    preds: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute classification metrics."""
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    avg = "macro"
    metrics = {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, average=avg, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, average=avg, zero_division=0)),
        "f1": float(f1_score(labels_arr, preds_arr, average=avg, zero_division=0)),
    }

    if class_names:
        report = classification_report(labels_arr, preds_arr, target_names=class_names, output_dict=True, zero_division=0)
        metrics["per_class"] = {
            cls: {k: v for k, v in vals.items() if k != "support"}
            for cls, vals in report.items()
            if cls in class_names
        }

    return metrics


def print_metrics(metrics: Dict) -> None:
    """Pretty-print metrics."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
    if "per_class" in metrics:
        print("\nPer-class breakdown:")
        for cls, vals in metrics["per_class"].items():
            print(f"  {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1-score']:.3f}")
    print("=" * 50 + "\n")
