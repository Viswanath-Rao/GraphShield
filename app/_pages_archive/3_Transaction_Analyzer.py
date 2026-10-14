"""
GraphShield Page 3: Real-Time Transaction Analyzer & Risk Scoring
"""

import streamlit as st
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import TransactionDatabase
from src.inference.predictor import FraudPredictor
from src.explainability.explainer import FraudExplainer
from src.graph.builder import FinancialGraphBuilder
from src.graph.analytics import GraphAnalyticsEngine
from app.components.network_graph import render_interactive_network

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚡ Real-Time Transaction Analyzer")
st.markdown("Score individual financial transactions, compute dynamic fraud risk ratings (0–100), and inspect factor attribution.")

db = TransactionDatabase()
predictor = FraudPredictor()
explainer = FraudExplainer()

# Selection mode
mode = st.radio("Input Source Mode:", ["Select from Database Records", "Hypothetical Scenario Simulator"], horizontal=True)

sample_tx = None

if mode == "Select from Database Records":
    df_samples = db.query_transactions("SELECT * FROM transactions ORDER BY tx_id DESC LIMIT 300")
    
    # Allow filtering sample pool
    tx_pool_choice = st.selectbox(
        "Choose Transaction to Inspect:",
        options=df_samples.index,
        format_func=lambda idx: f"ID #{df_samples.loc[idx, 'tx_id']} | Type: {df_samples.loc[idx, 'type']} | Amount: ${df_samples.loc[idx, 'amount']:,.2f} | Ground Truth: {'🚨 FRAUD' if df_samples.loc[idx, 'isFraud'] == 1 else '✅ LEGIT'}"
    )
    
    selected_row = df_samples.loc[tx_pool_choice]
    sample_tx = selected_row.to_dict()

else:
    st.subheader("Simulate Custom Financial Transaction")
    colA, colB, colC = st.columns(3)
    with colA:
        sim_type = st.selectbox("Transaction Modality", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        sim_amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=350000.0, step=5000.0)
        sim_step = st.number_input("Simulation Step (Hour)", min_value=1, max_value=744, value=48)
    with colB:
        sim_orig = st.text_input("Originator Account ID", value="C91827364")
        sim_old_orig = st.number_input("Originator Balance Pre-Tx ($)", min_value=0.0, value=350000.0, step=5000.0)
        sim_new_orig = st.number_input("Originator Balance Post-Tx ($)", min_value=0.0, value=0.0, step=5000.0)
    with colC:
        sim_dest = st.text_input("Beneficiary Account ID", value="C88889102")
        sim_old_dest = st.number_input("Beneficiary Balance Pre-Tx ($)", min_value=0.0, value=0.0, step=5000.0)
        sim_new_dest = st.number_input("Beneficiary Balance Post-Tx ($)", min_value=0.0, value=0.0, step=5000.0)

    sample_tx = {
        'tx_id': None,
        'step': int(sim_step),
        'type': sim_type,
        'amount': float(sim_amount),
        'nameOrig': sim_orig,
        'oldbalanceOrg': float(sim_old_orig),
        'newbalanceOrig': float(sim_new_orig),
        'nameDest': sim_dest,
        'oldbalanceDest': float(sim_old_dest),
        'newbalanceDest': float(sim_new_dest)
    }

st.markdown("---")

# Model selector for inference
col_m, col_btn = st.columns([3, 1])
with col_m:
    chosen_model = st.selectbox("Select Inference Model Architecture:", ["Random Forest", "GCN", "GAT", "Logistic Regression", "Gradient Boosting"])
with col_btn:
    st.write("")
    st.write("")
    run_scoring = st.button("🚀 Analyze Transaction", use_container_width=True)

if run_scoring or sample_tx is not None:
    # Run prediction
    try:
        prediction_res = predictor.predict_transaction(sample_tx, model_name=chosen_model, log_to_db=True)
        
        # Display Results
        risk_score = prediction_res['risk_score']
        risk_tier = prediction_res['risk_tier']
        decision = prediction_res['decision']
        
        # Determine styling
        tier_colors = {
            "CRITICAL": "#ef4444",
            "HIGH": "#f97316",
            "MEDIUM": "#f59e0b",
            "LOW": "#10b981"
        }
        color_hex = tier_colors.get(risk_tier, "#38bdf8")

        res_col1, res_col2, res_col3 = st.columns([2, 2, 3])
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {color_hex};">
                <div class="metric-title">Fraud Risk Score (0-100)</div>
                <div class="metric-value" style="color: {color_hex};">{risk_score:.1f} / 100</div>
                <div class="metric-badge badge-{risk_tier.lower()}">{risk_tier} RISK TIER</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Compliance Decision</div>
                <div class="metric-value" style="font-size: 1.4rem; color: #ffffff;">{decision}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">Estimated Fraud Prob: <strong>{prediction_res['fraud_probability']*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model Evaluation Rationale</div>
                <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.3rem;">
                    {prediction_res['justification']}
                </div>
                <div style="font-size: 0.75rem; color: #10b981; margin-top: 0.5rem;">✔ Persisted to SQLite Audit Trail</div>
            </div>
            """, unsafe_allow_html=True)

        # Engineered Signals Breakdown
        st.subheader("🔬 Derived Feature Signals")
        sig = prediction_res.get('engineered_signals', {})
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Originator Balance Error", f"${sig.get('errorBalanceOrig', 0.0):,.2f}")
        s2.metric("Beneficiary Balance Error", f"${sig.get('errorBalanceDest', 0.0):,.2f}")
        s3.metric("Drain Ratio (Amt/OldBal)", f"{sig.get('drainRatio', 0.0):.2f}")
        s4.metric("Complete Liquidation Flag", "YES (DRAINED)" if sig.get('isExactDrain') == 1 else "NO (PARTIAL)")

        # Local Factor Attribution
        st.subheader("📋 Explainability Factor Attribution")
        factors = explainer.explain_transaction_factors(sample_tx)
        for f in factors:
            f_color = "#ef4444" if f['severity'] == "CRITICAL" else ("#f97316" if f['severity'] == "HIGH" else "#38bdf8")
            st.markdown(f"""
            <div style="background: #162032; border-left: 3px solid {f_color}; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #ffffff;">{f['factor']}</strong>
                    <span style="color: {f_color}; font-weight: 600; font-family: 'JetBrains Mono';">{f['weight']}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">{f['description']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Graph Context Explorer
        st.subheader("🌐 Entity Graph Context")
        orig_acc = sample_tx['nameOrig']
        dest_acc = sample_tx['nameDest']
        st.write(f"Displaying interaction path between Originator `{orig_acc}` and Beneficiary `{dest_acc}`.")
        
        df_context = db.query_transactions(
            "SELECT * FROM transactions WHERE nameOrig = ? OR nameDest = ? OR nameOrig = ? OR nameDest = ? LIMIT 50",
            params=(orig_acc, orig_acc, dest_acc, dest_acc)
        )
        if not df_context.empty:
            builder = FinancialGraphBuilder()
            G_sub = builder.build_networkx_account_graph(df_context)
            fig_sub = render_interactive_network(G_sub, title=f"Entity Neighborhood: {orig_acc} ➔ {dest_acc}")
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            st.info("No prior connected transaction records for this entity pair in the current database sample.")

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")
