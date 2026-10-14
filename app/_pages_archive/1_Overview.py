"""
GraphShield Page 1: Overview & Executive Summary
"""

import streamlit as st
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import TransactionDatabase
from src.graph.builder import FinancialGraphBuilder
from app.components.metrics_cards import render_metric_card
from app.components.charts import plot_type_distribution, plot_amount_distribution
from app.components.network_graph import render_interactive_network

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🛡️ Executive Overview & System Status")
st.markdown("High-level system telemetry, transaction volume statistics, and graph topological properties.")

db = TransactionDatabase()
stats = db.get_summary_statistics()

# Row 1: KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Total Transactions", f"{stats.get('total_transactions', 0):,}", "In SQLite DB", "LIVE", "low")
with c2:
    render_metric_card("Total Volume", f"${stats.get('total_volume', 0.0):,.0f}", f"Avg ${stats.get('avg_amount', 0.0):,.2f}", "FINANCIAL", "low")
with c3:
    render_metric_card("Fraud Incidents", f"{stats.get('total_fraud', 0):,}", f"{stats.get('fraud_rate', 0.0):.2f}% Rate", "ALERT", "high")
with c4:
    render_metric_card("Monitored Accounts", f"{stats.get('total_accounts', 0):,}", "Originators & Beneficiaries", "ENTITIES", "low")
with c5:
    render_metric_card("Audit Records", f"{stats.get('total_audits', 0):,}", "Logged Predictions", "COMPLIANCE", "low")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Transaction Distribution by Modality")
    df = db.query_transactions("SELECT type, amount, isFraud FROM transactions LIMIT 5000")
    if not df.empty:
        fig_type = plot_type_distribution(df)
        st.plotly_chart(fig_type, use_container_width=True)

with col2:
    st.subheader("💰 Transaction Amount Density")
    if not df.empty:
        fig_amt = plot_amount_distribution(df)
        st.plotly_chart(fig_amt, use_container_width=True)

st.subheader("🌐 Financial Account Sub-Network Topology")
st.caption("Directed graph layout showing verified customers (cyan), merchant hubs (purple), and flagged high-risk entities (red).")
df_net = db.query_transactions("SELECT * FROM transactions LIMIT 1200")
builder = FinancialGraphBuilder()
G = builder.build_networkx_account_graph(df_net)
fig_net = render_interactive_network(G, title="Financial Account Transaction Graph", max_nodes=100)
st.plotly_chart(fig_net, use_container_width=True)
