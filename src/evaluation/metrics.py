"""
GraphShield Evaluation Metrics Module
Computes comprehensive fraud classification metrics: PR-AUC, ROC-AUC,
Recall, Precision, F1-Score, and Confusion Matrix diagnostics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger("GraphShield.Evaluation")

class ModelEvaluator:
    """Calculates standardized fraud evaluation metrics across tabular and GNN architectures."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Calculates all standard fraud detection metrics.
        Emphasizes PR-AUC, Recall, and F1 due to financial class imbalance.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        y_probs = np.asarray(y_probs).astype(float)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Handle edge cases for single-class subsets
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except ValueError:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except ValueError:
            pr_auc = float(y_true.mean())

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # ROC Curve coordinates
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        # Precision-Recall Curve coordinates
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)

        metrics = {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            },
            "pr_curve": {
                "precision": p_curve.tolist(),
                "recall": r_curve.tolist()
            }
        }

        logger.info("[%s] Evaluation -> PR-AUC: %.4f | ROC-AUC: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f",
                    model_name, pr_auc, roc_auc, rec, prec, f1)
        return metrics

    @staticmethod
    def create_comparison_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts a list of model metric dicts into a clean comparison DataFrame."""
        rows = []
        for m in metrics_list:
            rows.append({
                "Model": m["model_name"],
                "PR-AUC": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "F1-Score": m["f1_score"],
                "Recall": m["recall"],
                "Precision": m["precision"],
                "Accuracy": m["accuracy"]
            })
        df_comp = pd.DataFrame(rows).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
        return df_comp
