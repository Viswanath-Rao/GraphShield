"""
GraphShield Graph Attention Network (GAT) Module
Implements multi-head self-attention mechanisms to dynamically weigh
neighboring transactions and extract attention coefficients for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Optional

class FinancialGAT(nn.Module):
    """
    Graph Attention Network (GAT) for financial fraud detection.
    Computes masked self-attention coefficients to assign differential 
    importance to connected transactions (e.g. distinguishing normal high-frequency
    activity from coordinated smurfing and money mule cascades).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super(FinancialGAT, self).__init__()
        self.dropout = dropout

        # First GAT Layer with multi-head attention
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )
        self.norm1 = nn.BatchNorm1d(hidden_channels * heads)

        # Output GAT Layer: 1 attention head projecting to class logits
        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional attention weight retrieval for explainability."""
        # Layer 1
        if return_attention_weights:
            x, (edge_idx_att, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            alpha = None
            edge_idx_att = None

        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output Layer
        logits = self.gat2(x, edge_index)

        if return_attention_weights:
            return logits, (edge_idx_att, alpha)
        return logits

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculates softmax fraud probability distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, return_attention_weights=False)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1]
