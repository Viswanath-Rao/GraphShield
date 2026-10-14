"""
GraphShield Inference & Risk Scoring Engine
Executes real-time transaction scoring, model probability estimation,
risk categorization (0-100 scale), and audit logging.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.data.preprocessing import FinancialPreprocessor
from src.features.engineering import FinancialFeatureEngineer
from src.models.baselines import BaselineModelSuite
from src.models.gcn import FinancialGCN
from src.models.gat import FinancialGAT
from src.data.database import TransactionDatabase
import logging

logger = logging.getLogger("GraphShield.Predictor")

class FraudPredictor:
    """Provides inference, probability calibration, risk scoring, and audit logging."""

    def __init__(self, artifact_dir: Optional[str] = None, db_path: Optional[str] = None):
        if artifact_dir is None or artifact_dir == "models/" or artifact_dir == "models":
            if os.path.exists("models"):
                self.artifact_dir = "models"
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                self.artifact_dir = os.path.join(project_root, "models")
        else:
            self.artifact_dir = artifact_dir

        self.db = TransactionDatabase(db_path=db_path)
        self.engineer = FinancialFeatureEngineer()
        self.preprocessor = FinancialPreprocessor(artifact_dir=self.artifact_dir)
        self.baselines = BaselineModelSuite(model_dir=self.artifact_dir)

        # Lazy loaded PyTorch models
        self.gcn_model: Optional[FinancialGCN] = None
        self.gat_model: Optional[FinancialGAT] = None

    def predict_transaction(
        self,
        transaction_dict: Dict[str, Any],
        model_name: str = "Random Forest",
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a single transaction record through the complete inference pipeline:
        Feature engineering -> Preprocessing -> Model Scoring -> Risk Categorization -> Audit Log.
        """
        # 1. Input validation
        req_keys = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        for k in req_keys:
            if k not in transaction_dict:
                raise ValueError(f"Missing required transaction field: {k}")

        # 2. Feature Engineering & Preprocessing
        df_input = pd.DataFrame([transaction_dict])
        df_feat = self.engineer.engineer_features(df_input)
        X_vec = self.preprocessor.transform(df_feat)

        # 3. Model Inference
        fraud_prob = 0.0
        
        if model_name in self.baselines.models:
            fraud_prob = float(self.baselines.predict_proba(model_name, X_vec)[0])
        elif model_name == "GCN":
            self._ensure_gcn_loaded(in_channels=X_vec.shape[1])
            x_tensor = torch.tensor(X_vec, dtype=torch.float)
            # Standalone single node inference with self-loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            with torch.no_grad():
                out = self.gcn_model(x_tensor, edge_index)
                fraud_prob = float(torch.softmax(out, dim=1)[0, 1].item())
        elif model_name == "GAT":
            self._ensure_gat_loaded(in_channels=X_vec.shape[1])
            x_tensor = torch.tensor(X_vec, dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            with torch.no_grad():
                fraud_prob = float(self.gat_model.predict_proba(x_tensor, edge_index)[0].item())
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # 4. Risk Score (0 - 100 Scale)
        risk_score = round(float(fraud_prob * 100.0), 1)

        # 5. Risk Category & Decision
        if risk_score >= 85.0:
            risk_tier = "CRITICAL"
            decision = "BLOCKED"
        elif risk_score >= 60.0:
            risk_tier = "HIGH"
            decision = "FLAGGED FOR REVIEW"
        elif risk_score >= 30.0:
            risk_tier = "MEDIUM"
            decision = "ADDITIONAL AUTH REQUIRED"
        else:
            risk_tier = "LOW"
            decision = "APPROVED"

        # 6. Generate human-readable justification
        reasons = []
        amt = float(transaction_dict.get('amount', 0))
        old_o = float(transaction_dict.get('oldbalanceOrg', 0))
        new_o = float(transaction_dict.get('newbalanceOrig', 0))
        tx_type = str(transaction_dict.get('type', ''))

        if tx_type in ['TRANSFER', 'CASH_OUT']:
            reasons.append(f"High-risk transaction modality: {tx_type}")
        if old_o > 0 and new_o == 0:
            reasons.append("Account complete balance drain detected (zero residual balance)")
        if amt >= 200000:
            reasons.append(f"High-value transfer (${amt:,.2f}) exceeding compliance reporting threshold")
        if not reasons:
            reasons.append("Transaction characteristics align with legitimate financial baseline")

        justification = "; ".join(reasons)

        result = {
            "transaction_id": transaction_dict.get("tx_id", None),
            "model_used": model_name,
            "fraud_probability": round(fraud_prob, 4),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "decision": decision,
            "justification": justification,
            "engineered_signals": {
                "errorBalanceOrig": round(float(df_feat['errorBalanceOrig'].iloc[0]), 2),
                "errorBalanceDest": round(float(df_feat['errorBalanceDest'].iloc[0]), 2),
                "drainRatio": round(float(df_feat['drainRatio'].iloc[0]), 2),
                "isExactDrain": int(df_feat['isExactDrain'].iloc[0])
            }
        }

        # 7. Audit Logging to SQLite
        if log_to_db:
            try:
                self.db.log_audit_entry(
                    tx_id=transaction_dict.get("tx_id", None),
                    nameOrig=transaction_dict.get("nameOrig", "UNKNOWN"),
                    nameDest=transaction_dict.get("nameDest", "UNKNOWN"),
                    amount=amt,
                    model_used=model_name,
                    risk_score=risk_score,
                    prediction=decision,
                    justification=justification
                )
            except Exception as e:
                logger.warning("Could not write audit log to database: %s", e)

        return result

    def _ensure_gcn_loaded(self, in_channels: int):
        if self.gcn_model is None:
            self.gcn_model = FinancialGCN(in_channels=in_channels, hidden_channels=64, out_channels=2)
            weight_path = os.path.join(self.artifact_dir, "gcn_weights.pt")
            if os.path.exists(weight_path):
                self.gcn_model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
            self.gcn_model.eval()

    def _ensure_gat_loaded(self, in_channels: int):
        if self.gat_model is None:
            self.gat_model = FinancialGAT(in_channels=in_channels, hidden_channels=32, out_channels=2, heads=4)
            weight_path = os.path.join(self.artifact_dir, "gat_weights.pt")
            if os.path.exists(weight_path):
                self.gat_model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
            self.gat_model.eval()
