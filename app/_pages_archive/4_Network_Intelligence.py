"""
GraphShield Page 4: Network Intelligence & Graph Analytics
"""

import streamlit as st
import os
import sys
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import TransactionDatabase
from src.graph.builder import FinancialGraphBuilder
from src.graph.analytics import GraphAnalyticsEngine
from app.components.network_graph import render_interactive_network
from app.components.metrics_cards import render_metric_card

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🕸️ Network Intelligence & Entity Graphs")
st.markdown("Analyze financial graph topologies, detect money mule syndicates, and explore local k-hop neighborhoods.")

db = TransactionDatabase()

# Fetch transactions to construct graph
df_tx = db.query_transactions("SELECT * FROM transactions LIMIT 3000")
builder = FinancialGraphBuilder()
G = builder.build_networkx_account_graph(df_tx, max_sample_rows=2500)

stats = GraphAnalyticsEngine.compute_summary_statistics(G)

# Network KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Network Nodes", f"{stats['num_nodes']:,}", "Active Accounts", "TOPOLOGY", "low")
with c2:
    render_metric_card("Directed Edges", f"{stats['num_edges']:,}", "Transaction Paths", "FLOWS", "low")
with c3:
    render_metric_card("Graph Density", f"{stats['density']:.6f}", "Connectivity ratio", "SPARSE", "low")
with c4:
    render_metric_card("Components", f"{stats['connected_components']:,}", "Sub-clusters", "CLUSTERS", "low")
with c5:
    render_metric_card("Fraud Flagged Nodes", f"{stats['fraud_nodes']:,}", "Involved Accounts", "SUSPICIOUS", "high" if stats['fraud_nodes'] > 0 else "low")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🌐 Full Network Graph", "🚨 Money Mule Ring Detection", "🔍 Ego-Network Explorer"])

with tab1:
    st.subheader("Interactive Financial Topology View")
    node_limit = st.slider("Display Node Limit", min_value=30, max_value=250, value=100, step=10)
    fig_network = render_interactive_network(G, title=f"Account Interaction Topology ({node_limit} Nodes)", max_nodes=node_limit)
    st.plotly_chart(fig_network, use_container_width=True)

with tab2:
    st.subheader("Identified Money Mule Rings & Syndicate Hubs")
    st.caption("Accounts flagged by GraphShield's topological heuristic: high in-degree, pass-through volume conduit, and fraud transaction links.")
    
    mules = GraphAnalyticsEngine.identify_suspicious_mule_rings(G, top_k=15)
    if mules:
        df_mules = pd.DataFrame(mules)
        st.dataframe(
            df_mules.style.format({
                "in_volume": "${:,.2f}",
                "out_volume": "${:,.2f}",
                "suspicion_score": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No high-degree mule rings identified in this sample window.")

with tab3:
    st.subheader("Account Ego-Network Neighborhood Explorer")
    st.caption("Isolate a specific account node and inspect its 1-hop or 2-hop transaction network.")
    
    all_nodes = list(G.nodes())
    if all_nodes:
        # Preselect a mule node if available
        default_node = mules[0]['account_id'] if mules else all_nodes[0]
        selected_account = st.selectbox("Select Account ID to Inspect:", options=all_nodes, index=all_nodes.index(default_node))
        hop_radius = st.radio("Neighborhood Radius:", [1, 2], horizontal=True)

        sub_G = GraphAnalyticsEngine.extract_ego_subgraph(G, node_id=selected_account, radius=hop_radius)
        st.write(f"Ego-Network contains **{sub_G.number_of_nodes()}** connected entities and **{sub_G.number_of_edges()}** transactions.")
        
        fig_ego = render_interactive_network(sub_G, title=f"Ego-Network: {selected_account} ({hop_radius}-Hop Radius)")
        st.plotly_chart(fig_ego, use_container_width=True)
