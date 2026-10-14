"""
GraphShield Graph Analytics Module
Calculates network topology metrics, degree distributions, connected components,
and discovers high-risk money mule rings and suspicious transaction clusters.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("GraphShield.GraphAnalytics")

class GraphAnalyticsEngine:
    """Analyzes topology, flow dynamics, and suspicious structures in financial graphs."""

    @staticmethod
    def compute_summary_statistics(G: nx.DiGraph) -> Dict[str, Any]:
        """Calculates global network statistics."""
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes == 0:
            return {
                "num_nodes": 0, "num_edges": 0, "density": 0.0,
                "avg_degree": 0.0, "connected_components": 0, "fraud_nodes": 0
            }

        density = nx.density(G)
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        avg_degree = (sum(in_degrees) + sum(out_degrees)) / (2.0 * num_nodes)

        # Connected components (weakly connected for directed graphs)
        components = nx.number_weakly_connected_components(G)
        
        # Fraud node count
        fraud_nodes = sum(1 for _, data in G.nodes(data=True) if data.get('risk') == 'Fraudulent')

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": round(float(density), 6),
            "avg_degree": round(float(avg_degree), 3),
            "connected_components": components,
            "fraud_nodes": fraud_nodes,
            "legit_nodes": num_nodes - fraud_nodes,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0
        }

    @staticmethod
    def identify_suspicious_mule_rings(G: nx.DiGraph, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Detects potential money mule accounts based on:
        - High degree centrality (hubs receiving from multiple senders)
        - Pass-through flow (inflow and outflow presence)
        - Association with flagged transactions
        """
        mules = []
        for node, data in G.nodes(data=True):
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            fraud_count = data.get('fraud_count', 0)
            
            # Calculate total incoming and outgoing volume
            in_vol = sum(edge_data.get('amount', 0.0) for _, _, edge_data in G.in_edges(node, data=True))
            out_vol = sum(edge_data.get('amount', 0.0) for _, _, edge_data in G.out_edges(node, data=True))
            
            # Suspicion score: composite metric of in-degree, volume, and fraud links
            suspicion_score = (in_deg * 2.0) + (out_deg * 1.5) + (fraud_count * 15.0)
            if in_vol > 0 and out_vol > 0:
                suspicion_score += 10.0  # Pass-through conduit bonus

            if in_deg + out_deg > 1 or fraud_count > 0:
                mules.append({
                    "account_id": node,
                    "account_type": data.get("node_type", "Customer"),
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "in_volume": round(in_vol, 2),
                    "out_volume": round(out_vol, 2),
                    "fraud_count": fraud_count,
                    "suspicion_score": round(suspicion_score, 1)
                })

        mules.sort(key=lambda x: x['suspicion_score'], reverse=True)
        return mules[:top_k]

    @staticmethod
    def extract_ego_subgraph(G: nx.DiGraph, node_id: str, radius: int = 1) -> nx.DiGraph:
        """Extracts a local k-hop neighborhood graph around a specific account."""
        if not G.has_node(node_id):
            return nx.DiGraph()
        
        # Convert to undirected to get 360-degree neighbors, then induce subgraph
        undirected_G = G.to_undirected()
        nodes_in_radius = nx.single_source_shortest_path_length(undirected_G, node_id, cutoff=radius).keys()
        subgraph = G.subgraph(nodes_in_radius).copy()
        return subgraph
