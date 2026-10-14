"""
Tests for GraphShield Inference Engine & Explainability
"""

import pytest
from src.inference.predictor import FraudPredictor
from src.explainability.explainer import FraudExplainer

def test_inference_pipeline():
    predictor = FraudPredictor()
    
    test_tx = {
        'tx_id': 9999,
        'step': 24,
        'type': 'TRANSFER',
        'amount': 450000.0,
        'nameOrig': 'C81726354',
        'oldbalanceOrg': 450000.0,
        'newbalanceOrig': 0.0,
        'nameDest': 'C12345678',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0
    }

    # Test baseline inference
    res_rf = predictor.predict_transaction(test_tx, model_name="Random Forest", log_to_db=False)
    assert 0.0 <= res_rf['risk_score'] <= 100.0
    assert res_rf['risk_tier'] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert "engineered_signals" in res_rf
    assert res_rf['engineered_signals']['isExactDrain'] == 1

def test_explainer_factors():
    explainer = FraudExplainer()
    test_tx = {
        'type': 'TRANSFER',
        'amount': 300000.0,
        'oldbalanceOrg': 300000.0,
        'newbalanceOrig': 0.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0
    }
    factors = explainer.explain_transaction_factors(test_tx)
    assert len(factors) >= 2
    factor_names = [f['factor'] for f in factors]
    assert "Total Account Liquidation" in factor_names
