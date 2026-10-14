"""
GraphShield Page 7: SQL Workbench & Data Explorer
Provides live SQL query execution, table inspection, and CSV export.
"""

import streamlit as st
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.database import TransactionDatabase

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "../styles/custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🗄️ SQL Workbench & Database Explorer")
st.markdown("Direct SQL querying against GraphShield's embedded SQLite database (`transactions.db`).")

db = TransactionDatabase()

tab1, tab2 = st.tabs(["💻 Interactive SQL Query Workbench", "📋 Relational Table Browser"])

with tab1:
    st.subheader("SQL Terminal & Query Executor")
    st.caption("Write and execute custom SQL statements against `transactions`, `accounts`, `audit_logs`, or `model_benchmarks`.")

    # Preset query shortcuts
    presets = {
        "Volume & Fraud Breakdown by Modality": """
SELECT 
    type,
    COUNT(*) AS total_transactions,
    SUM(amount) AS total_volume_usd,
    AVG(amount) AS avg_amount_usd,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) * 100.0 / COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY type
ORDER BY fraud_count DESC;
        """.strip(),
        
        "Top 10 High-Risk Accounts by Fraud Activity": """
SELECT 
    account_id,
    account_type,
    total_sent,
    total_received,
    out_degree,
    in_degree,
    fraud_involvement,
    risk_tier
FROM accounts
ORDER BY fraud_involvement DESC, total_sent DESC
LIMIT 10;
        """.strip(),

        "High-Value Transfers Exceeding $250,000": """
SELECT 
    tx_id, step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, isFraud
FROM transactions
WHERE amount >= 250000.0
ORDER BY amount DESC
LIMIT 15;
        """.strip(),

        "Latest Real-Time Prediction Audit Logs": """
SELECT 
    log_id, nameOrig, nameDest, amount, model_used, risk_score, prediction, justification, evaluated_at
FROM audit_logs
ORDER BY log_id DESC
LIMIT 15;
        """.strip()
    }

    selected_preset = st.selectbox("Load Sample SQL Query Preset:", ["Custom Query"] + list(presets.keys()))
    
    initial_sql = presets[selected_preset] if selected_preset != "Custom Query" else "SELECT * FROM transactions LIMIT 25;"
    
    user_sql = st.text_area("SQL Statement (SELECT only):", value=initial_sql, height=140)
    
    col_run, col_clear = st.columns([1, 5])
    with col_run:
        exec_btn = st.button("▶ Run Query", type="primary", use_container_width=True)

    if exec_btn:
        if not user_sql.strip().upper().startswith("SELECT"):
            st.warning("Only SELECT statements are permitted in the public query workbench.")
        else:
            try:
                df_result = db.execute_custom_query(user_sql)
                st.success(f"Query returned {len(df_result):,} rows.")
                st.dataframe(df_result, use_container_width=True)

                # CSV Download
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Query Result as CSV",
                    data=csv_data,
                    file_name="graphshield_query_result.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"SQL Execution Error: {str(e)}")

with tab2:
    st.subheader("Database Schema & Raw Tables")
    table_choice = st.selectbox("Select Table to Inspect:", ["transactions", "accounts", "audit_logs", "model_benchmarks"])
    
    page_size = st.slider("Page Size (Rows):", min_value=10, max_value=200, value=25, step=5)
    page_num = st.number_input("Page Number:", min_value=1, value=1, step=1)
    offset = (page_num - 1) * page_size

    df_table = db.query_transactions(f"SELECT * FROM {table_choice} LIMIT {page_size} OFFSET {offset}")
    st.dataframe(df_table, use_container_width=True)
    st.caption(f"Displaying rows {offset + 1} to {offset + len(df_table)} of table `{table_choice}`.")
