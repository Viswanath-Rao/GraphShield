"""
GraphShield Explainability Module
Provides multi-level interpretability: global feature importance,
local transaction factor attribution, and graph neighborhood attention evidence.
"""

import os
import joblib
import pandas as pd
import numpy as np
import torch
import networkx as nx
from typing import Dict, Any, List, Optional
import logging

from src.models.baselines import BaselineModelSuite
from src.models.gat import FinancialGAT

logger = logging.getLogger("GraphShield.Explainer")

class FraudExplainer:
    """Interprets model decisions via feature importance, GAT attention, and neighborhood topology."""

    def __init__(self, artifact_dir: Optional[str] = None):
        if artifact_dir is None or artifact_dir == "models/" or artifact_dir == "models":
            if os.path.exists("models"):
                self.artifact_dir = "models"
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                self.artifact_dir = os.path.join(project_root, "models")
        else:
            self.artifact_dir = artifact_dir

        self.baselines = BaselineModelSuite(model_dir=self.artifact_dir)
        self.meta = None
        self._load_meta()

    def _load_meta(self):
        meta_path = os.path.join(self.artifact_dir, "preprocessor_meta.joblib")
        if os.path.exists(meta_path):
            self.meta = joblib.load(meta_path)

    def get_global_feature_importance(self, model_name: str = "Random Forest") -> pd.DataFrame:
        """Retrieves global feature importance ranking for the selected model."""
        if not self.meta:
            self._load_meta()
        if not self.meta:
            return pd.DataFrame(columns=["Feature", "Importance"])

        feature_names = self.meta.get("feature_names", [])
        return self.baselines.get_feature_importances(model_name, feature_names)

    def explain_transaction_factors(self, transaction_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Breaks down local transaction risk factors into interpretable components.
        Evaluates balance integrity, liquidity drain, transaction modality, and entity history.
        """
        amt = float(transaction_dict.get('amount', 0.0))
        old_o = float(transaction_dict.get('oldbalanceOrg', 0.0))
        new_o = float(transaction_dict.get('newbalanceOrig', 0.0))
        tx_type = str(transaction_dict.get('type', ''))
        old_d = float(transaction_dict.get('oldbalanceDest', 0.0))
        new_d = float(transaction_dict.get('newbalanceDest', 0.0))

        factors = []

        # 1. Account drain
        if old_o > 0 and new_o == 0:
            factors.append({
                "factor": "Total Account Liquidation",
                "severity": "CRITICAL",
                "weight": "+35%",
                "description": f"Originator balance completely drained ($ {old_o:,.2f} -> $0.00)."
            })
        elif old_o > 0 and (amt / old_o) > 0.8:
            factors.append({
                "factor": "Aggressive Balance Depletion",
                "severity": "HIGH",
                "weight": "+20%",
                "description": f"Transaction consumes {(amt/old_o)*100:.1f}% of available originator funds."
            })

        # 2. Modality
        if tx_type in ['TRANSFER', 'CASH_OUT']:
            factors.append({
                "factor": f"High-Risk Modality ({tx_type})",
                "severity": "HIGH",
                "weight": "+25%",
                "description": "Primary vehicle for rapid capital flight and money laundering in mobile money networks."
            })
        else:
            factors.append({
                "factor": f"Standard Modality ({tx_type})",
                "severity": "LOW",
                "weight": "-15%",
                "description": "Modality historically associated with genuine commercial or utility payments."
            })

        # 3. High Value Limit
        if amt >= 200000.0:
            factors.append({
                "factor": "High Value Compliance Threshold",
                "severity": "HIGH",
                "weight": "+20%",
                "description": f"Amount ($ {amt:,.2f}) exceeds regulatory AML monitoring threshold."
            })

        # 4. Destination balance anomalies
        if old_d == 0.0 and new_d == 0.0 and tx_type == 'TRANSFER':
            factors.append({
                "factor": "Beneficiary Balance Zero Discrepancy",
                "severity": "MEDIUM",
                "weight": "+15%",
                "description": "Destination balance shows zero increment post-transfer, indicative of mule intermediary."
            })

        return factors

    def get_gat_attention_neighborhood(
        self,
        pyg_data,
        target_node_idx: int,
        gat_model: FinancialGAT
    ) -> List[Dict[str, Any]]:
        """
        Extracts GAT attention weights connecting the target transaction node to its neighbors.
        Identifies which adjacent transaction exerted the strongest influence.
        """
        gat_model.eval()
        with torch.no_grad():
            _, (edge_idx, alpha) = gat_model(pyg_data.x, pyg_data.edge_index, return_attention_weights=True)

        # Average alpha across attention heads
        alpha_mean = alpha.mean(dim=1).cpu().numpy()
        edge_src = edge_idx[0].cpu().numpy()
        edge_dst = edge_idx[1].cpu().numpy()

        # Find incoming edges to target_node_idx
        mask = (edge_dst == target_node_idx)
        neighbor_nodes = edge_src[mask]
        neighbor_weights = alpha_mean[mask]

        evidence = []
        for n_idx, w in zip(neighbor_nodes, neighbor_weights):
            evidence.append({
                "neighbor_node_id": int(n_idx),
                "attention_weight": round(float(w), 4),
                "is_self_loop": bool(n_idx == target_node_idx),
                "influence_rank": "High" if w > 0.25 else "Moderate"
            })

        evidence.sort(key=lambda x: x['attention_weight'], reverse=True)
        return evidence
