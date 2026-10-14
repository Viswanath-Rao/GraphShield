"""
GraphShield Feature Engineering Module
Derives critical financial domain features, balance error discrepancies,
temporal cycles, and behavioral statistics without label leakage.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger("GraphShield.Features")

class FinancialFeatureEngineer:
    """Computes domain-specific fraud indicators and financial behavioral metrics."""

    def __init__(self):
        self.engineered_feature_names: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates financial fraud signals:
        1. Balance calculation discrepancies (errorBalanceOrig, errorBalanceDest)
        2. Draining ratio (amount / oldbalanceOrg)
        3. Temporal cyclical patterns (hour_of_day, day_of_week)
        4. Entity type flags (is_merchant_dest)
        5. Round number structuring flags
        6. Network activity counters
        """
        df = df.copy()

        # 1. Balance discrepancy features (crucial PaySim signature)
        # For a clean transfer: oldbalanceOrg - amount == newbalanceOrig => error == 0
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

        # 2. Complete drain ratio
        df['drainRatio'] = np.where(
            df['oldbalanceOrg'] > 0,
            np.clip(df['amount'] / (df['oldbalanceOrg'] + 1.0), 0.0, 10.0),
            0.0
        )
        # Exact drain indicator (account completely emptied)
        df['isExactDrain'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(int)

        # 3. Temporal cycles
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        df['day_of_week'] = (df['step'] // 24) % 7

        # Cyclical sin/cos encodings for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)

        # 4. Entity category flag
        df['is_merchant_dest'] = df['nameDest'].astype(str).str.startswith('M').astype(int)

        # 5. Financial structuring indicators (amounts close to reporting limits)
        df['is_round_10k'] = ((df['amount'] % 10000 == 0) & (df['amount'] > 0)).astype(int)
        df['is_high_value'] = (df['amount'] >= 200000.0).astype(int)

        # 6. Entity activity counts within the window
        orig_counts = df['nameOrig'].value_counts().to_dict()
        dest_counts = df['nameDest'].value_counts().to_dict()
        df['orig_degree'] = df['nameOrig'].map(orig_counts).fillna(1).astype(int)
        df['dest_degree'] = df['nameDest'].map(dest_counts).fillna(1).astype(int)

        # Record feature column names
        derived_cols = [
            'errorBalanceOrig', 'errorBalanceDest', 'drainRatio', 'isExactDrain',
            'hour_of_day', 'hour_sin', 'hour_cos', 'day_of_week',
            'is_merchant_dest', 'is_round_10k', 'is_high_value',
            'orig_degree', 'dest_degree'
        ]
        self.engineered_feature_names = derived_cols
        logger.info("Engineered %s financial features successfully.", len(derived_cols))

        return df
