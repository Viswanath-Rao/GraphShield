"""
GraphShield Page 8: About & System Information
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("ℹ️ About GraphShield & Engineering Architecture")
st.markdown("Academic Capstone Project & Portfolio Specification for B.Tech Computer Science and Engineering (Machine Learning).")

st.markdown("""
<div class="glass-panel" style="line-height: 1.7; color: #cbd5e1;">
    <h3 style="color: #38bdf8; margin-top: 0;">Project Abstract</h3>
    <p>
        <strong>GraphShield</strong> is an advanced financial cybersecurity intelligence platform designed to detect fraudulent transactions 
        and money laundering rings by fusing relational feature engineering with <strong>Graph Neural Networks (GNNs)</strong>.
    </p>
    <p>
        Conventional fraud scoring engines analyze transactions in isolation, leaving institutions vulnerable to coordinated money mule syndicates, 
        smurfing (structuring), and multi-hop account takeover drains. GraphShield models financial interactions as a dynamic directed graph, 
        leveraging <strong>Graph Convolutional Networks (GCN)</strong> for topological feature smoothing and 
        <strong>Graph Attention Networks (GAT)</strong> for differential edge attention attribution.
    </p>
</div>
""", unsafe_allow_html=True)

st.subheader("🛠️ Full Technology Stack Breakdown")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Core & Deep Learning:**
    - `Python 3.14 / 3.11+`
    - `PyTorch 2.x` (Deep Learning Tensor Engine)
    - `PyTorch Geometric (PyG)` (GCN & GAT Layers)
    - `NetworkX 3.x` (Graph Theory & Topologies)
    """)

with col2:
    st.markdown("""
    **Data & Persistence:**
    - `SQLite 3` (Relational Audit & Fast Querying)
    - `SQL Engine` (Custom Query Workbench)
    - `Pandas & NumPy` (Data Manipulation)
    - `Scikit-Learn` (Baselines & Metrics)
    """)

with col3:
    st.markdown("""
    **UI & Visualization:**
    - `Streamlit` (Web Intelligence Dashboard)
    - `Plotly Express / Graph Objects` (Interactive Visuals)
    - `Custom CSS` (Cybersecurity Theme)
    - `Pytest` (Automated Reliability Suite)
    """)

st.subheader("📚 Dataset Provenance & Attribution")
st.markdown("""
<div class="glass-panel" style="font-size: 0.95rem; color: #94a3b8; line-height: 1.6;">
    <strong>Benchmark Dataset:</strong> PaySim Mobile Money Simulation Dataset<br>
    <strong>Primary Citation:</strong> E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, 
    <em>"PaySim: A financial mobile money simulator for fraud detection,"</em> 
    in European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus, 2016.<br>
    <strong>License / Usage:</strong> Public Academic Research & Demonstration License.
</div>
""", unsafe_allow_html=True)

st.subheader("⚠️ Academic Disclaimer")
st.info("""
GraphShield is an educational, research, and technical portfolio demonstration platform developed for 
a Bachelor of Technology (B.Tech) degree in Computer Science and Engineering with specialization in Machine Learning. 
Risk scores and probabilities represent algorithmic model outputs and should be used alongside qualified human compliance review in production banking environments.
""")
