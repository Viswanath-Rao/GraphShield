"""
GraphShield Page 5: Model Intelligence & GNN Comparative Benchmarks
"""

import streamlit as st
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.evaluation.metrics import ModelEvaluator
from app.components.charts import (
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix, plot_training_history
)

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Model Intelligence & Benchmarking")
st.markdown("Rigorous evaluation comparing traditional machine learning baselines against Graph Neural Networks (GCN, GAT).")

metrics_file = os.path.join("models", "metrics.json")
if not os.path.exists(metrics_file):
    st.error("Model metrics not found. Please execute the trainer pipeline.")
    st.stop()

with open(metrics_file) as f:
    metrics = json.load(f)

# Model Comparison Table
st.subheader("📊 Architectural Performance Leaderboard")
st.caption("Evaluated on strictly held-out test splits without temporal or label leakage.")

rows = []
for name, m in metrics.items():
    rows.append({
        "Model": name,
        "PR-AUC": m.get("pr_auc", 0.0),
        "ROC-AUC": m.get("roc_auc", 0.0),
        "Recall": m.get("recall", 0.0),
        "Precision": m.get("precision", 0.0),
        "F1-Score": m.get("f1_score", 0.0),
        "Accuracy": m.get("accuracy", 0.0)
    })

df_leaderboard = pd.DataFrame(rows).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

st.dataframe(
    df_leaderboard.style.format({
        "PR-AUC": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "Recall": "{:.4f}",
        "Precision": "{:.4f}",
        "F1-Score": "{:.4f}",
        "Accuracy": "{:.4f}"
    }).highlight_max(subset=["PR-AUC", "Recall", "F1-Score"], color="#1e3a8a"),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📉 ROC & PR Curves", "🔲 Confusion Matrices", "🔄 GNN Training History", "⚖️ Trade-off Analysis"])

with tab1:
    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.subheader("ROC Curves")
        fig_roc = plot_roc_curves(metrics)
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_pr:
        st.subheader("Precision-Recall Curves")
        fig_pr = plot_pr_curves(metrics)
        st.plotly_chart(fig_pr, use_container_width=True)

with tab2:
    st.subheader("Confusion Matrix Diagnostics")
    selected_model = st.selectbox("Select Model to Inspect Confusion Matrix:", list(metrics.keys()))
    cm = metrics[selected_model].get("confusion_matrix", {})
    fig_cm = plot_confusion_matrix(cm, selected_model)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # False positive / negative metrics
    c_fn, c_fp = st.columns(2)
    c_fn.metric("False Negatives (Missed Fraud)", f"{cm.get('false_negative', 0):,}", "High Cost: Direct Financial Loss", delta_color="inverse")
    c_fp.metric("False Positives (False Alarms)", f"{cm.get('false_positive', 0):,}", "Customer Friction Cost", delta_color="inverse")

with tab3:
    st.subheader("Graph Neural Network Loss Convergence")
    gnn_choice = st.radio("Choose GNN Architecture:", ["GCN", "GAT"], horizontal=True)
    if gnn_choice in metrics and "training_history" in metrics[gnn_choice]:
        fig_hist = plot_training_history(metrics[gnn_choice]["training_history"], gnn_choice)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Training history not available for this model.")

with tab4:
    st.subheader("Practical Engineering Trade-Offs")
    st.markdown("""
    <div class="glass-panel" style="font-size: 0.95rem; line-height: 1.7; color: #cbd5e1;">
        <h4>Tabular Baselines vs. Graph Neural Networks in Production:</h4>
        <ul>
            <li><strong>Random Forest:</strong> High tabular precision and lightning-fast sub-millisecond inference per transaction. Highly effective for single-point liquidation detection, but blind to multi-hop mule networks.</li>
            <li><strong>Graph Convolutional Network (GCN):</strong> Aggregates k-hop neighborhood structures, effectively detecting transaction clusters. Requires sub-graph mini-batching for latency-sensitive payment gateways.</li>
            <li><strong>Graph Attention Network (GAT):</strong> Highest representational power by assigning differential attention weights to suspicious edges while attenuating noisy high-volume legitimate transactions. Essential for money laundering syndicates.</li>
            <li><strong>Evaluation Imbalance Focus:</strong> Due to extreme class imbalance (fraud ~ 3.5%), accuracy is a deceptive metric. <strong>PR-AUC and Recall</strong> serve as the primary engineering optimization targets.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
