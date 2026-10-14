"""
Tests for GraphShield SQLite Database Module
"""

import pytest
import os
import pandas as pd
from src.data.database import TransactionDatabase

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_transactions.db"
    return TransactionDatabase(db_path=str(db_file))

def test_database_initialization(test_db):
    conn = test_db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "transactions" in tables
    assert "accounts" in tables
    assert "audit_logs" in tables
    assert "model_benchmarks" in tables

def test_insert_and_query_transactions(test_db):
    sample_data = pd.DataFrame([{
        'step': 1,
        'type': 'TRANSFER',
        'amount': 50000.0,
        'nameOrig': 'C10001',
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 0.0,
        'nameDest': 'C20002',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFraud': 1,
        'isFlaggedFraud': 0
    }])
    count = test_db.insert_transactions_df(sample_data)
    assert count == 1

    df_out = test_db.query_transactions("SELECT * FROM transactions WHERE nameOrig = 'C10001'")
    assert len(df_out) == 1
    assert df_out.iloc[0]['amount'] == 50000.0

def test_audit_logging(test_db):
    log_id = test_db.log_audit_entry(
        tx_id=99,
        nameOrig="C111",
        nameDest="C222",
        amount=12000.0,
        model_used="Random Forest",
        risk_score=85.5,
        prediction="BLOCKED",
        justification="High risk drain"
    )
    assert log_id > 0
    logs = test_db.get_audit_logs(limit=10)
    assert len(logs) == 1
    assert logs.iloc[0]['risk_score'] == 85.5
