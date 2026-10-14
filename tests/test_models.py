"""
Tests for GraphShield Model Architectures (Baselines, GCN, GAT)
"""

import pytest
import torch
import numpy as np
from src.models.gcn import FinancialGCN
from src.models.gat import FinancialGAT
from src.models.baselines import BaselineModelSuite

def test_gcn_forward_pass():
    in_channels = 16
    num_nodes = 20
    x = torch.randn(num_nodes, in_channels)
    # Simple ring edge index
    sources = list(range(num_nodes))
    targets = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    model = FinancialGCN(in_channels=in_channels, hidden_channels=32, out_channels=2)
    logits = model(x, edge_index)
    assert logits.shape == (num_nodes, 2)

    probs = model.predict_proba(x, edge_index)
    assert probs.shape == (num_nodes,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_gat_forward_pass():
    in_channels = 16
    num_nodes = 15
    x = torch.randn(num_nodes, in_channels)
    sources = list(range(num_nodes))
    targets = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    model = FinancialGAT(in_channels=in_channels, hidden_channels=16, out_channels=2, heads=2)
    logits, (edge_att, alpha) = model(x, edge_index, return_attention_weights=True)

    assert logits.shape == (num_nodes, 2)
    assert alpha is not None
    assert alpha.dim() == 2

def test_baseline_models_training(tmp_path):
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, size=50)

    suite = BaselineModelSuite(model_dir=str(tmp_path))
    suite.train_all(X, y)

    for name in suite.models:
        assert suite.is_trained[name] is True
        probs = suite.predict_proba(name, X[:5])
        assert len(probs) == 5
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
