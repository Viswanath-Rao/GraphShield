"""
Tests for GraphShield Data Loader Module
"""

import pytest
import os
import pandas as pd
from src.data.loader import FinancialDataLoader, REQUIRED_COLUMNS

def test_benchmark_dataset_generation(tmp_path):
    csv_file = str(tmp_path / "test_data.csv")
    loader = FinancialDataLoader(raw_path=csv_file)
    
    df = loader.generate_benchmark_dataset(n_samples=500, fraud_ratio=0.05, save_path=csv_file)
    assert len(df) == 500
    assert os.path.exists(csv_file)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert df['isFraud'].sum() > 0

def test_schema_validation(tmp_path):
    csv_file = str(tmp_path / "valid_data.csv")
    loader = FinancialDataLoader(raw_path=csv_file)
    df = loader.generate_benchmark_dataset(n_samples=200, save_path=csv_file)
    
    summary = loader.validate_schema(df)
    assert summary['total_records'] == 200
    assert summary['fraud_records'] >= 0

def test_schema_validation_missing_columns():
    loader = FinancialDataLoader()
    invalid_df = pd.DataFrame({'step': [1], 'amount': [100.0]})
    with pytest.raises(ValueError):
        loader.validate_schema(invalid_df)
