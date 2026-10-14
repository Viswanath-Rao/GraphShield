"""
Tests for GraphShield Graph Construction and Analytics
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from src.graph.builder import FinancialGraphBuilder
from src.graph.analytics import GraphAnalyticsEngine

@pytest.fixture
def sample_graph_df():
    return pd.DataFrame([
        {'step': 1, 'type': 'TRANSFER', 'amount': 10000.0, 'nameOrig': 'C1', 'oldbalanceOrg': 10000.0, 'newbalanceOrig': 0.0, 'nameDest': 'C2', 'oldbalanceDest': 0.0, 'newbalanceDest': 10000.0, 'isFraud': 1},
        {'step': 2, 'type': 'CASH_OUT', 'amount': 10000.0, 'nameOrig': 'C2', 'oldbalanceOrg': 10000.0, 'newbalanceOrig': 0.0, 'nameDest': 'M1', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0, 'isFraud': 1},
        {'step': 3, 'type': 'PAYMENT', 'amount': 50.0, 'nameOrig': 'C3', 'oldbalanceOrg': 500.0, 'newbalanceOrig': 450.0, 'nameDest': 'M1', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0, 'isFraud': 0}
    ])

def test_networkx_graph_builder(sample_graph_df):
    builder = FinancialGraphBuilder()
    G = builder.build_networkx_account_graph(sample_graph_df)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 4  # C1, C2, M1, C3
    assert G.number_of_edges() == 3
    assert G.has_edge('C1', 'C2')
    assert G.has_edge('C2', 'M1')

def test_graph_analytics_engine(sample_graph_df):
    builder = FinancialGraphBuilder()
    G = builder.build_networkx_account_graph(sample_graph_df)
    
    stats = GraphAnalyticsEngine.compute_summary_statistics(G)
    assert stats['num_nodes'] == 4
    assert stats['num_edges'] == 3
    assert stats['fraud_nodes'] >= 2

    # Mule ring detection (C2 is an intermediary conduit)
    mules = GraphAnalyticsEngine.identify_suspicious_mule_rings(G)
    assert len(mules) > 0
    mule_ids = [m['account_id'] for m in mules]
    assert 'C2' in mule_ids
