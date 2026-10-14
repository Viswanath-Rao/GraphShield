"""
GraphShield Preprocessing Module
Handles feature encoding, normalization, dataset splitting, and artifact serialization.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger("GraphShield.Preprocessing")

class FinancialPreprocessor:
    """Standardizes, encodes, and splits financial transaction records."""

    def __init__(self, artifact_dir: str = "models/"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        self.all_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        self.type_encoder.fit(np.array(self.all_types).reshape(-1, 1))

        self.numerical_cols = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'errorBalanceOrig', 'errorBalanceDest', 'drainRatio', 'isExactDrain',
            'hour_sin', 'hour_cos', 'is_merchant_dest', 'is_round_10k', 'is_high_value',
            'orig_degree', 'dest_degree'
        ]
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fits scaler and encodes features for the full dataset."""
        df_encoded = self._encode_categorical(df)
        
        # Fit scaler on numerical features
        num_data = df[self.numerical_cols].fillna(0.0).values
        num_scaled = self.scaler.fit_transform(num_data)

        type_encoded = df_encoded.values
        X = np.hstack([num_scaled, type_encoded])
        
        type_feature_names = [f"type_{t}" for t in self.type_encoder.categories_[0]]
        self.feature_names = self.numerical_cols + type_feature_names
        self.is_fitted = True

        y = df['isFraud'].values.astype(int)

        self.save_artifacts()
        logger.info("Preprocessor fitted. Total features: %d", len(self.feature_names))
        return X, y, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms new input records using already-fitted parameters."""
        if not self.is_fitted:
            self.load_artifacts()

        df_encoded = self._encode_categorical(df)
        num_data = df[self.numerical_cols].fillna(0.0).values
        num_scaled = self.scaler.transform(num_data)
        type_encoded = df_encoded.values

        X = np.hstack([num_scaled, type_encoded])
        return X

    def transform_single_record(self, record_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single transaction dictionary into the exact feature vector for inference."""
        from src.features.engineering import FinancialFeatureEngineer
        fe = FinancialFeatureEngineer()
        
        temp_df = pd.DataFrame([record_dict])
        temp_df = fe.engineer_features(temp_df)
        return self.transform(temp_df)

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes the transaction type."""
        types = df[['type']].copy()
        encoded = self.type_encoder.transform(types.values)
        cols = [f"type_{t}" for t in self.type_encoder.categories_[0]]
        return pd.DataFrame(encoded, columns=cols, index=df.index)

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates stratified train, validation, and test splits preventing data leakage."""
        # First split into train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split train_val into train and validation
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_size, stratify=y_train_val, random_state=random_state
        )

        logger.info("Splits created: Train=%d (fraud=%d), Val=%d (fraud=%d), Test=%d (fraud=%d)",
                    len(y_train), int(y_train.sum()),
                    len(y_val), int(y_val.sum()),
                    len(y_test), int(y_test.sum()))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_artifacts(self) -> None:
        """Serializes the scaler, encoder, and feature names."""
        scaler_path = os.path.join(self.artifact_dir, "scaler.joblib")
        encoder_path = os.path.join(self.artifact_dir, "type_encoder.joblib")
        meta_path = os.path.join(self.artifact_dir, "preprocessor_meta.joblib")

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.type_encoder, encoder_path)
        joblib.dump({
            "numerical_cols": self.numerical_cols,
            "feature_names": self.feature_names
        }, meta_path)
        logger.info("Preprocessor artifacts saved to %s", self.artifact_dir)

    def load_artifacts(self) -> None:
        """Loads serialized preprocessor artifacts."""
        scaler_path = os.path.join(self.artifact_dir, "scaler.joblib")
        encoder_path = os.path.join(self.artifact_dir, "type_encoder.joblib")
        meta_path = os.path.join(self.artifact_dir, "preprocessor_meta.joblib")

        if not os.path.exists(scaler_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Preprocessor artifacts not found in models/ directory. Run training first.")

        self.scaler = joblib.load(scaler_path)
        self.type_encoder = joblib.load(encoder_path)
        meta = joblib.load(meta_path)
        self.numerical_cols = meta["numerical_cols"]
        self.feature_names = meta["feature_names"]
        self.is_fitted = True
        logger.info("Preprocessor artifacts loaded successfully.")
