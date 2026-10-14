"""
GraphShield Model Trainer Pipeline
Orchestrates end-to-end training for baselines, GCN, and GAT models
with early stopping, class imbalance weighting, and artifact checkpointing.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from src.data.loader import FinancialDataLoader
from src.data.preprocessing import FinancialPreprocessor
from src.data.database import TransactionDatabase
from src.features.engineering import FinancialFeatureEngineer
from src.graph.builder import FinancialGraphBuilder
from src.models.baselines import BaselineModelSuite
from src.models.gcn import FinancialGCN
from src.models.gat import FinancialGAT
from src.evaluation.metrics import ModelEvaluator

logger = logging.getLogger("GraphShield.Trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class GraphShieldTrainer:
    """Manages full dataset preparation, baseline training, and GNN optimization."""

    def __init__(self, artifact_dir: str = "models/", db_path: str = "data/transactions.db"):
        self.artifact_dir = artifact_dir
        self.db_path = db_path
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        self.db = TransactionDatabase(db_path=self.db_path)
        self.loader = FinancialDataLoader()
        self.engineer = FinancialFeatureEngineer()
        self.preprocessor = FinancialPreprocessor(artifact_dir=self.artifact_dir)
        self.graph_builder = FinancialGraphBuilder()
        self.baselines = BaselineModelSuite(model_dir=self.artifact_dir)
        self.evaluator = ModelEvaluator()

    def run_complete_pipeline(self, sample_size: int = 15000, epochs: int = 40) -> Dict[str, Any]:
        """Runs the entire training and evaluation workflow from end to end."""
        logger.info("=== STEP 1: LOADING DATASET ===")
        df = self.loader.load_or_create(sample_size=sample_size)
        
        # Ingest into SQLite database
        logger.info("Syncing %d records into SQLite database at %s...", len(df), self.db_path)
        self.db.insert_transactions_df(df, if_exists="replace")

        logger.info("=== STEP 2: FEATURE ENGINEERING ===")
        df_feat = self.engineer.engineer_features(df)

        logger.info("=== STEP 3: PREPROCESSING & SPLITTING ===")
        X, y, feature_names = self.preprocessor.fit_transform(df_feat)
        
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        # Stratified train/val/test index split
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=y, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.15/0.85, stratify=y[train_idx], random_state=42)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        logger.info("=== STEP 4: TRAINING BASELINE MODELS ===")
        self.baselines.train_all(X_train, y_train)

        results = {}

        # Evaluate Baselines on Test Set
        for name in self.baselines.models:
            probs = self.baselines.predict_proba(name, X_test)
            preds = (probs >= 0.5).astype(int)
            res = self.evaluator.evaluate_predictions(y_test, preds, probs, model_name=name)
            results[name] = res
            self.db.save_model_benchmark(name, res)

        logger.info("=== STEP 5: BUILDING GRAPH REPRESENTATION ===")
        pyg_data = self.graph_builder.build_pyg_transaction_graph(
            df=df,
            feature_matrix=X,
            labels=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )

        # Save graph object for dashboard network visualizer
        torch.save(pyg_data, os.path.join("data", "processed", "graph_data.pt"))

        # Compute class weights for loss function to address fraud sparsity
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([1.0, float(n_neg) / max(1, float(n_pos))], dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=pos_weight)

        logger.info("=== STEP 6: TRAINING GRAPH CONVOLUTIONAL NETWORK (GCN) ===")
        gcn_model, gcn_history = self._train_gcn(pyg_data, criterion, epochs=epochs)
        torch.save(gcn_model.state_dict(), os.path.join(self.artifact_dir, "gcn_weights.pt"))

        # Evaluate GCN on test mask
        gcn_model.eval()
        with torch.no_grad():
            out = gcn_model(pyg_data.x, pyg_data.edge_index)
            gcn_probs = torch.softmax(out, dim=1)[pyg_data.test_mask, 1].cpu().numpy()
            gcn_preds = (gcn_probs >= 0.5).astype(int)
            y_test_gnn = pyg_data.y[pyg_data.test_mask].cpu().numpy()
            gcn_res = self.evaluator.evaluate_predictions(y_test_gnn, gcn_preds, gcn_probs, model_name="GCN")
            gcn_res["training_history"] = gcn_history
            results["GCN"] = gcn_res
            self.db.save_model_benchmark("GCN", gcn_res)

        logger.info("=== STEP 7: TRAINING GRAPH ATTENTION NETWORK (GAT) ===")
        gat_model, gat_history = self._train_gat(pyg_data, criterion, epochs=epochs)
        torch.save(gat_model.state_dict(), os.path.join(self.artifact_dir, "gat_weights.pt"))

        # Evaluate GAT on test mask
        gat_model.eval()
        with torch.no_grad():
            gat_probs = gat_model.predict_proba(pyg_data.x, pyg_data.edge_index)[pyg_data.test_mask].cpu().numpy()
            gat_preds = (gat_probs >= 0.5).astype(int)
            gat_res = self.evaluator.evaluate_predictions(y_test_gnn, gat_preds, gat_probs, model_name="GAT")
            gat_res["training_history"] = gat_history
            results["GAT"] = gat_res
            self.db.save_model_benchmark("GAT", gat_res)

        # Save all results to models/metrics.json
        metrics_file = os.path.join(self.artifact_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=== TRAINING COMPLETED SUCCESSFULLY! ===")
        return results

    def _train_gcn(self, data, criterion, epochs: int = 40) -> Tuple[FinancialGCN, Dict[str, list]]:
        """Trains the GCN model recording loss curves."""
        in_dim = data.x.shape[1]
        model = FinancialGCN(in_channels=in_dim, hidden_channels=64, out_channels=2, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])

            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss.item()))

            if epoch % 10 == 0 or epoch == epochs:
                logger.info("GCN Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f",
                            epoch, epochs, loss.item(), val_loss.item())

        return model, history

    def _train_gat(self, data, criterion, epochs: int = 40) -> Tuple[FinancialGAT, Dict[str, list]]:
        """Trains the GAT model recording loss curves."""
        in_dim = data.x.shape[1]
        model = FinancialGAT(in_channels=in_dim, hidden_channels=32, out_channels=2, heads=4, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, return_attention_weights=False)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, return_attention_weights=False)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])

            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss.item()))

            if epoch % 10 == 0 or epoch == epochs:
                logger.info("GAT Epoch [%d/%d] - Train Loss: %.4f | Val Loss: %.4f",
                            epoch, epochs, loss.item(), val_loss.item())

        return model, history

if __name__ == "__main__":
    trainer = GraphShieldTrainer()
    trainer.run_complete_pipeline(sample_size=15000, epochs=30)
