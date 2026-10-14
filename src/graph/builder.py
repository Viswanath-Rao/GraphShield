"""
GraphShield Graph Builder Module
Constructs both PyTorch Geometric graph data objects (for GNN node classification)
and NetworkX directed multigraphs (for account relationship analytics and visualization).
"""

import torch
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
import logging

logger = logging.getLogger("GraphShield.GraphBuilder")

class FinancialGraphBuilder:
    """Constructs graph representations for financial transactions and account networks."""

    def __init__(self, max_edges_per_node: int = 10):
        self.max_edges_per_node = max_edges_per_node

    def build_pyg_transaction_graph(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> Data:
        """
        Constructs a PyTorch Geometric Data object where:
        - Nodes represent individual transactions.
        - Edges connect transactions sharing the same originator or beneficiary entity.
        - Node features x are the engineered financial attributes.
        - Labels y indicate whether a transaction is fraudulent.
        """
        num_nodes = len(df)
        x = torch.tensor(feature_matrix, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Build shared-entity edges efficiently
        edge_sources = []
        edge_targets = []

        # Group by nameOrig and nameDest to connect related transactions
        for entity_col in ['nameOrig', 'nameDest']:
            entity_groups = df.groupby(entity_col).indices
            for entity, indices in entity_groups.items():
                if len(indices) > 1:
                    # Limit connections per entity to avoid dense clique explosion
                    indices = indices[:self.max_edges_per_node]
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            u, v = indices[i], indices[j]
                            edge_sources.extend([u, v])
                            edge_targets.extend([v, u])

        if len(edge_sources) == 0:
            # Fallback self-loops if no shared entities exist in tiny sample
            edge_sources = list(range(num_nodes))
            edge_targets = list(range(num_nodes))

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Train, val, test masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes
        )

        logger.info("Constructed PyG Transaction Graph: %d nodes, %d edges.", num_nodes, edge_index.shape[1])
        return data

    def build_networkx_account_graph(self, df: pd.DataFrame, max_sample_rows: int = 2000) -> nx.DiGraph:
        """
        Constructs a directed NetworkX graph representing the account-to-account payment network:
        - Nodes = Account IDs (Customer/Merchant)
        - Edges = Transactions (directed from Originator to Destination)
        """
        sample_df = df.head(max_sample_rows) if len(df) > max_sample_rows else df
        G = nx.DiGraph()

        for _, row in sample_df.iterrows():
            orig = str(row['nameOrig'])
            dest = str(row['nameDest'])
            amount = float(row['amount'])
            is_fraud = int(row['isFraud'])
            tx_type = str(row['type'])
            step = int(row['step'])

            # Add nodes with initial attributes
            if not G.has_node(orig):
                G.add_node(orig, node_type="Customer", risk="Fraudulent" if is_fraud else "Normal", fraud_count=0)
            if not G.has_node(dest):
                dest_type = "Merchant" if dest.startswith("M") else "Customer"
                G.add_node(dest, node_type=dest_type, risk="Fraudulent" if is_fraud else "Normal", fraud_count=0)

            if is_fraud:
                G.nodes[orig]['risk'] = "Fraudulent"
                G.nodes[orig]['fraud_count'] += 1
                G.nodes[dest]['risk'] = "Fraudulent"
                G.nodes[dest]['fraud_count'] += 1

            # Add directed edge
            G.add_edge(
                orig, dest,
                amount=amount,
                is_fraud=is_fraud,
                tx_type=tx_type,
                step=step
            )

        logger.info("Constructed NetworkX Account Graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
        return G
