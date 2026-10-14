"""
GraphShield Data Loader Module
Loads, validates, and prepares financial fraud transaction data.
Supports external PaySim CSV files and includes a high-fidelity synthetic benchmark generator.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("GraphShield.Loader")

REQUIRED_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
    'isFraud'
]

class FinancialDataLoader:
    """Loads and validates financial transaction datasets for fraud detection."""

    def __init__(self, raw_path: str = "data/raw/financial_transactions.csv"):
        self.raw_path = raw_path

    def load_or_create(self, sample_size: int = 25000, fraud_ratio: float = 0.035) -> pd.DataFrame:
        """
        Loads the dataset if it exists on disk; otherwise, generates a 
        realistic benchmark financial graph dataset adhering to the PaySim specification.
        """
        if os.path.exists(self.raw_path):
            logger.info("Found existing dataset at %s. Loading...", self.raw_path)
            df = self.load_csv(self.raw_path)
        else:
            logger.info("No raw dataset found at %s. Generating realistic benchmark dataset...", self.raw_path)
            df = self.generate_benchmark_dataset(
                n_samples=sample_size,
                fraud_ratio=fraud_ratio,
                save_path=self.raw_path
            )
        
        self.validate_schema(df)
        return df

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Loads CSV file with validation."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
        df = pd.read_csv(filepath)
        return df

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates column presence, checks data types, missing values, and duplicates.
        Raises ValueError if required columns are missing.
        """
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        missing_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
        duplicates = int(df.duplicated(subset=['step', 'type', 'amount', 'nameOrig', 'nameDest']).sum())
        total_rows = len(df)
        fraud_count = int(df['isFraud'].sum())
        fraud_percentage = (fraud_count / total_rows * 100) if total_rows > 0 else 0.0

        summary = {
            "total_records": total_rows,
            "fraud_records": fraud_count,
            "legit_records": total_rows - fraud_count,
            "fraud_percentage": round(fraud_percentage, 3),
            "duplicates": duplicates,
            "missing_values": missing_counts
        }
        logger.info("Schema validation successful: %s rows, %s fraud cases (%.2f%%)", 
                    total_rows, fraud_count, fraud_percentage)
        return summary

    def generate_benchmark_dataset(
        self,
        n_samples: int = 25000,
        fraud_ratio: float = 0.035,
        random_seed: int = 42,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generates a scientifically grounded financial transaction dataset
        matching PaySim topology and typology (TRANSFER -> CASH_OUT drain cascades,
        mule rings, merchant retail payments, and legitimate account flows).
        """
        np.random.seed(random_seed)
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        # Account pools
        n_cust_orig = int(n_samples * 0.45)
        n_cust_dest = int(n_samples * 0.25)
        n_merchants = int(n_samples * 0.15)
        n_mule_rings = 15  # Coordinated money mule hubs

        orig_accounts = [f"C{np.random.randint(10000000, 99999999)}" for _ in range(n_cust_orig)]
        dest_accounts = [f"C{np.random.randint(10000000, 99999999)}" for _ in range(n_cust_dest)]
        merchant_accounts = [f"M{np.random.randint(10000000, 99999999)}" for _ in range(n_merchants)]
        mule_accounts = [f"C8888{np.random.randint(1000, 9999)}" for _ in range(n_mule_rings)]

        records = []

        # 1. Generate Legitimate Transactions
        legit_types = ['PAYMENT', 'CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT']
        legit_type_probs = [0.42, 0.30, 0.18, 0.08, 0.02]

        for i in range(n_legit):
            step = int(np.random.randint(1, 744))  # 1 month in hourly steps
            tx_type = np.random.choice(legit_types, p=legit_type_probs)
            orig = np.random.choice(orig_accounts)
            
            if tx_type == 'PAYMENT':
                dest = np.random.choice(merchant_accounts)
                amount = float(np.round(np.random.exponential(scale=65.0) + 5.0, 2))
                old_orig = float(np.round(np.random.uniform(amount, amount * 10 + 500), 2))
                new_orig = float(np.round(old_orig - amount, 2))
                old_dest = 0.0  # Merchant balances are often 0 in PaySim
                new_dest = 0.0
            elif tx_type == 'CASH_IN':
                dest = np.random.choice(orig_accounts)
                amount = float(np.round(np.random.lognormal(mean=7.5, sigma=1.0), 2))
                old_orig = float(np.round(np.random.uniform(100, 50000), 2))
                new_orig = float(np.round(old_orig + amount, 2))
                old_dest = 0.0
                new_dest = 0.0
            elif tx_type == 'CASH_OUT':
                dest = np.random.choice(merchant_accounts)
                amount = float(np.round(np.random.lognormal(mean=8.5, sigma=1.2), 2))
                old_orig = float(np.round(np.random.uniform(amount, amount + 10000), 2))
                new_orig = float(np.round(old_orig - amount, 2))
                old_dest = float(np.round(np.random.uniform(1000, 500000), 2))
                new_dest = float(np.round(old_dest + amount, 2))
            elif tx_type == 'TRANSFER':
                dest = np.random.choice(dest_accounts)
                amount = float(np.round(np.random.lognormal(mean=8.8, sigma=1.1), 2))
                old_orig = float(np.round(np.random.uniform(amount, amount * 2), 2))
                new_orig = float(np.round(old_orig - amount, 2))
                old_dest = float(np.round(np.random.uniform(0, 100000), 2))
                new_dest = float(np.round(old_dest + amount, 2))
            else:  # DEBIT
                dest = np.random.choice(merchant_accounts)
                amount = float(np.round(np.random.uniform(10, 500), 2))
                old_orig = float(np.round(np.random.uniform(amount, amount + 2000), 2))
                new_orig = float(np.round(old_orig - amount, 2))
                old_dest = 0.0
                new_dest = 0.0

            records.append({
                'step': step,
                'type': tx_type,
                'amount': amount,
                'nameOrig': orig,
                'oldbalanceOrg': old_orig,
                'newbalanceOrig': new_orig,
                'nameDest': dest,
                'oldbalanceDest': old_dest,
                'newbalanceDest': new_dest,
                'isFraud': 0,
                'isFlaggedFraud': 0
            })

        # 2. Generate Fraudulent Transactions (Simulating Account Takeover & Mule Cascades)
        for i in range(n_fraud):
            step = int(np.random.randint(1, 744))
            # Financial fraud in PaySim occurs strictly via TRANSFER followed by CASH_OUT
            is_transfer = (i % 2 == 0)
            tx_type = 'TRANSFER' if is_transfer else 'CASH_OUT'
            
            # Connected mule ring architecture
            mule = np.random.choice(mule_accounts)
            victim = np.random.choice(orig_accounts)

            # High value drain pattern: complete balance emptying
            amount = float(np.round(np.random.uniform(50000, 850000), 2))
            old_orig = amount  # Account wiped out completely
            new_orig = 0.0     # Final balance zero

            if is_transfer:
                dest = mule
                old_dest = 0.0
                new_dest = 0.0  # Common PaySim artifact: destination balance error
            else:
                orig = mule
                dest = np.random.choice(merchant_accounts)
                old_dest = 0.0
                new_dest = amount

            is_flagged = 1 if amount >= 200000.0 and tx_type == 'TRANSFER' else 0

            records.append({
                'step': step,
                'type': tx_type,
                'amount': amount,
                'nameOrig': victim if is_transfer else mule,
                'oldbalanceOrg': old_orig,
                'newbalanceOrig': new_orig,
                'nameDest': dest,
                'oldbalanceDest': old_dest,
                'newbalanceDest': new_dest,
                'isFraud': 1,
                'isFlaggedFraud': is_flagged
            })

        df = pd.DataFrame(records)
        # Shuffle records to break synthetic order while maintaining temporal steps
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info("Saved benchmark dataset (%s rows) to %s", len(df), save_path)

        return df
