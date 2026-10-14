"""
GraphShield Page 2: Fraud Analytics & Exploratory Intelligence
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import TransactionDatabase
from app.components.charts import plot_hourly_pattern, plot_type_distribution, THEME_LAYOUT

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📈 Fraud Analytics & Pattern Exploration")
st.markdown("Investigate financial fraud distributions, temporal spikes, and balance liquidation signatures.")

db = TransactionDatabase()

# Fetch data for interactive exploration
df = db.query_transactions("SELECT * FROM transactions LIMIT 10000")

# Sidebar Filters
st.sidebar.header("🔍 Analytical Filters")
all_types = list(df['type'].unique())
selected_types = st.sidebar.multiselect("Transaction Types", all_types, default=all_types)
fraud_filter = st.sidebar.radio("Fraud Filter", ["All Transactions", "Fraudulent Only", "Legitimate Only"])

min_amt, max_amt = float(df['amount'].min()), float(df['amount'].max())
amt_range = st.sidebar.slider("Amount Range ($)", min_amt, max_amt, (min_amt, max_amt))

# Apply filters
filtered_df = df[df['type'].isin(selected_types)]
filtered_df = filtered_df[(filtered_df['amount'] >= amt_range[0]) & (filtered_df['amount'] <= amt_range[1])]

if fraud_filter == "Fraudulent Only":
    filtered_df = filtered_df[filtered_df['isFraud'] == 1]
elif fraud_filter == "Legitimate Only":
    filtered_df = filtered_df[filtered_df['isFraud'] == 0]

# Metrics banner
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Records", f"{len(filtered_df):,}")
col2.metric("Filtered Volume", f"${filtered_df['amount'].sum():,.2f}")
f_count = int(filtered_df['isFraud'].sum())
col3.metric("Fraud Count", f"{f_count:,}")
f_pct = (f_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0.0
col4.metric("Fraud Rate", f"{f_pct:.2f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🕒 Temporal Patterns", "⚖️ Balance Anomalies", "📊 Modality Breakdown"])

with tab1:
    st.subheader("24-Hour Fraud Frequency Distribution")
    st.caption("Visualizes transaction concentration across diurnal cycles (step % 24).")
    if not filtered_df.empty:
        fig_hour = plot_hourly_pattern(filtered_df)
        st.plotly_chart(fig_hour, use_container_width=True)

with tab2:
    st.subheader("Originator Balance Depletion Signature")
    st.caption("Fraudulent transfers (red) typically originate along the line `newbalanceOrig == 0` (complete account drain).")
    if not filtered_df.empty:
        sample_scatter = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)
        sample_scatter['Status'] = sample_scatter['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        fig_scatter = px.scatter(
            sample_scatter,
            x='oldbalanceOrg',
            y='newbalanceOrig',
            color='Status',
            size='amount',
            color_discrete_map={'Legitimate': '#06b6d4', 'Fraudulent': '#ef4444'},
            labels={'oldbalanceOrg': 'Initial Balance ($)', 'newbalanceOrig': 'New Balance ($)'},
            title="Pre-Transaction vs Post-Transaction Originator Balance"
        )
        fig_scatter.update_layout(**THEME_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Transaction Modality Comparison")
    if not filtered_df.empty:
        fig_type = plot_type_distribution(filtered_df)
        st.plotly_chart(fig_type, use_container_width=True)

st.markdown("""
<div class="glass-panel" style="margin-top: 1rem;">
    <h4 style="color: #38bdf8; margin-top: 0;">💡 Key Empirical Observations</h4>
    <ul style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        <li><strong>Modality Concentration:</strong> Financial fraud in mobile money simulations is concentrated exclusively in <code>TRANSFER</code> (initial theft) and <code>CASH_OUT</code> (capital extraction).</li>
        <li><strong>Liquidation Signature:</strong> Over 95% of confirmed fraud attempts result in an exact <code>newbalanceOrig == 0</code>, reflecting unauthorized automated sweeps.</li>
        <li><strong>Structuring:</strong> High-value transfers frequently target round numbers or amounts just below standard monitoring thresholds.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
