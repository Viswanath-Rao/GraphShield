"""
GraphShield Baseline Machine Learning Models
Implements Logistic Regression, Random Forest, and Gradient Boosting benchmarks
with class-imbalance compensation, probability calibration, and feature importance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("GraphShield.Baselines")

class BaselineModelSuite:
    """Manages training, evaluation, and persistence of tabular ML baseline models."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            "Gradient Boosting": HistGradientBoostingClassifier(
                max_iter=100,
                class_weight='balanced',
                random_state=42
            )
        }
        self.is_trained: Dict[str, bool] = {name: False for name in self.models}

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains all baseline models on the designated training dataset."""
        for name, model in self.models.items():
            logger.info("Training %s baseline model...", name)
            model.fit(X_train, y_train)
            self.is_trained[name] = True
            logger.info("%s training complete.", name)
        
        self.save_models()

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability of fraud (class 1) for the requested model."""
        if not self.is_trained.get(model_name, False):
            self.load_model(model_name)
        
        model = self.models[model_name]
        probs = model.predict_proba(X)
        return probs[:, 1]

    def predict(self, model_name: str, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions using configurable decision threshold."""
        probs = self.predict_proba(model_name, X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """Returns feature importance ranking for compatible models."""
        if not self.is_trained.get(model_name, False):
            self.load_model(model_name)

        model = self.models[model_name]
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is not None and len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
            return df_imp
        
        return pd.DataFrame(columns=["Feature", "Importance"])

    def save_models(self) -> None:
        """Serializes all trained models to disk."""
        for name, model in self.models.items():
            if self.is_trained.get(name, False):
                filename = f"baseline_{name.lower().replace(' ', '_')}.joblib"
                filepath = os.path.join(self.model_dir, filename)
                joblib.dump(model, filepath)
                logger.info("Saved %s to %s", name, filepath)

    def load_model(self, model_name: str) -> None:
        """Loads a specific model from disk."""
        filename = f"baseline_{model_name.lower().replace(' ', '_')}.joblib"
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved artifact for {model_name} not found at {filepath}")
        
        self.models[model_name] = joblib.load(filepath)
        self.is_trained[model_name] = True
        logger.info("Loaded %s from %s", model_name, filepath)

    def load_all(self) -> bool:
        """Attempts to load all baseline models from disk. Returns True if all exist."""
        all_found = True
        for name in self.models:
            filename = f"baseline_{name.lower().replace(' ', '_')}.joblib"
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                self.is_trained[name] = True
            else:
                all_found = False
        return all_found
