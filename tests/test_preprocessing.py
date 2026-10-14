"""
Tests for GraphShield Feature Engineering and Preprocessing
"""

import pytest
import numpy as np
import pandas as pd
from src.features.engineering import FinancialFeatureEngineer
from src.data.preprocessing import FinancialPreprocessor

@pytest.fixture
def sample_raw_df():
    return pd.DataFrame([
        {
            'step': 12,
            'type': 'TRANSFER',
            'amount': 250000.0,
            'nameOrig': 'C1001',
            'oldbalanceOrg': 250000.0,
            'newbalanceOrig': 0.0,
            'nameDest': 'C2002',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'isFraud': 1,
            'isFlaggedFraud': 1
        },
        {
            'step': 14,
            'type': 'PAYMENT',
            'amount': 45.0,
            'nameOrig': 'C3003',
            'oldbalanceOrg': 1000.0,
            'newbalanceOrig': 955.0,
            'nameDest': 'M4004',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0,
            'isFraud': 0,
            'isFlaggedFraud': 0
        }
    ])

def test_feature_engineering_signals(sample_raw_df):
    fe = FinancialFeatureEngineer()
    df_feat = fe.engineer_features(sample_raw_df)

    assert 'errorBalanceOrig' in df_feat.columns
    assert 'drainRatio' in df_feat.columns
    assert 'isExactDrain' in df_feat.columns
    
    # First row is exact drain: oldbalanceOrg == 250000, newbalanceOrig == 0
    assert df_feat.loc[0, 'isExactDrain'] == 1
    assert df_feat.loc[0, 'drainRatio'] > 0.99
    # Second row is payment to merchant
    assert df_feat.loc[1, 'is_merchant_dest'] == 1

def test_preprocessing_fit_transform(tmp_path, sample_raw_df):
    fe = FinancialFeatureEngineer()
    df_feat = fe.engineer_features(sample_raw_df)
    
    preprocessor = FinancialPreprocessor(artifact_dir=str(tmp_path))
    X, y, names = preprocessor.fit_transform(df_feat)

    assert X.shape[0] == 2
    assert len(y) == 2
    assert len(names) == X.shape[1]
    assert np.all(np.isfinite(X))
