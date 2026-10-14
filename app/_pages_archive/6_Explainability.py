"""
GraphShield Page 6: Explainability & Interpretability Center
"""

import streamlit as st
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.explainability.explainer import FraudExplainer
from src.data.database import TransactionDatabase
from app.components.charts import plot_feature_importance

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🔍 Model Explainability & AML Audit Center")
st.markdown("Deconstruct black-box machine learning and GNN predictions into compliant, human-auditable risk explanations.")

explainer = FraudExplainer()
db = TransactionDatabase()

tab1, tab2, tab3 = st.tabs(["📊 Global Feature Importance", "🎯 Local Transaction Attribution", "⚡ GAT Attention Dynamics"])

with tab1:
    st.subheader("Global Tabular Feature Attribution")
    st.caption("Derived from Mean Decrease in Impurity (MDI) across ensemble decision trees.")
    
    top_n = st.slider("Select Number of Top Features:", min_value=5, max_value=20, value=10)
    df_imp = explainer.get_global_feature_importance("Random Forest")
    
    if not df_imp.empty:
        fig_imp = plot_feature_importance(df_imp, top_n=top_n)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        with st.expander("View Full Feature Ranking Table"):
            st.dataframe(df_imp, use_container_width=True)
    else:
        st.warning("Feature importance metadata not found. Run training pipeline.")

with tab2:
    st.subheader("Local Risk Factor Dissection")
    st.caption("Inspect how specific features contributed positively or negatively to an individual transaction's risk score.")
    
    # Query suspicious transaction sample
    df_sample = db.query_transactions("SELECT * FROM transactions WHERE isFraud = 1 LIMIT 20")
    if not df_sample.empty:
        selected_idx = st.selectbox(
            "Select Flagged Transaction for Audit:",
            options=df_sample.index,
            format_func=lambda idx: f"ID #{df_sample.loc[idx, 'tx_id']} | Type: {df_sample.loc[idx, 'type']} | Amount: ${df_sample.loc[idx, 'amount']:,.2f} | Orig: {df_sample.loc[idx, 'nameOrig']}"
        )
        tx = df_sample.loc[selected_idx].to_dict()
        factors = explainer.explain_transaction_factors(tx)

        for f in factors:
            border_col = "#ef4444" if f['severity'] == "CRITICAL" else ("#f97316" if f['severity'] == "HIGH" else "#06b6d4")
            st.markdown(f"""
            <div style="background: #162032; border-left: 4px solid {border_col}; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between;">
                    <h4 style="margin: 0; color: #ffffff;">{f['factor']}</h4>
                    <span style="color: {border_col}; font-family: 'JetBrains Mono'; font-weight: 700;">{f['weight']}</span>
                </div>
                <p style="margin: 0.4rem 0 0 0; color: #94a3b8; font-size: 0.9rem;">{f['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No flagged transactions found in database.")

with tab3:
    st.subheader("Graph Attention Network (GAT) Edge Weight Dynamics")
    st.markdown(r"""
    <div class="glass-panel" style="font-size: 0.95rem; line-height: 1.7; color: #cbd5e1;">
        <h4 style="color: #38bdf8; margin-top: 0;">How GAT Attends to Financial Relationships:</h4>
        <p>
            In traditional GCNs, message aggregation is normalized purely based on node degrees:
            \[ h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}} W^{(l)} h_j^{(l)} \right) \]
            This treats all neighbors equally regardless of their transaction amount, balance drain, or fraud propensity.
        </p>
        <p>
            In <strong>GraphShield's GAT</strong>, the model computes self-attention coefficients across multiple attention heads (\(K=4\)):
            \[ \alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \parallel W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \parallel W h_k]\right)\right)} \]
            <strong>Practical Impact:</strong> High attention weights (\(\alpha_{ij} > 0.35\)) are dynamically assigned to edges connecting money mule accounts, while benign merchant edges are suppressed (\(\alpha_{ij} < 0.05\)).
        </p>
    </div>
    """, unsafe_allow_html=True)
