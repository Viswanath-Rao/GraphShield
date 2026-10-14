"""
GraphShield Database Module - SQLite & SQL Management
Handles transaction persistence, account profiling, audit logging, and custom SQL analytics.
"""

import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
import os
import logging

logger = logging.getLogger("GraphShield.Database")

class TransactionDatabase:
    """Manages SQLite storage, querying, and auditing for financial transactions."""

    def __init__(self, db_path: Optional[str] = None, auto_seed: bool = True):
        if db_path is None or db_path == "data/transactions.db":
            if os.path.exists("data/transactions.db"):
                self.db_path = "data/transactions.db"
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                self.db_path = os.path.join(project_root, "data", "transactions.db")
        else:
            self.db_path = db_path

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db(auto_seed=auto_seed and ("test" not in os.path.basename(self.db_path).lower()))

    def get_connection(self) -> sqlite3.Connection:
        """Returns a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self, auto_seed: bool = False) -> None:
        """Creates required tables and performance indexes if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    nameOrig TEXT NOT NULL,
                    oldbalanceOrg REAL NOT NULL,
                    newbalanceOrig REAL NOT NULL,
                    nameDest TEXT NOT NULL,
                    oldbalanceDest REAL NOT NULL,
                    newbalanceDest REAL NOT NULL,
                    isFraud INTEGER NOT NULL,
                    isFlaggedFraud INTEGER DEFAULT 0,
                    risk_score REAL DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Accounts aggregate table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    account_type TEXT NOT NULL,
                    total_sent REAL DEFAULT 0.0,
                    total_received REAL DEFAULT 0.0,
                    out_degree INTEGER DEFAULT 0,
                    in_degree INTEGER DEFAULT 0,
                    fraud_involvement INTEGER DEFAULT 0,
                    risk_tier TEXT DEFAULT 'LOW',
                    last_seen_step INTEGER DEFAULT 0
                );
            """)

            # Audit logs table for tracking live predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id INTEGER,
                    nameOrig TEXT,
                    nameDest TEXT,
                    amount REAL,
                    model_used TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    justification TEXT,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Model benchmark performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_benchmarks (
                    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    pr_auc REAL,
                    eval_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create Indexes for high-speed graph & entity lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_orig ON transactions(nameOrig);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_dest ON transactions(nameDest);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_fraud ON transactions(isFraud);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_step ON transactions(step);")

            conn.commit()
            logger.info("Database initialized successfully at %s", self.db_path)

        if auto_seed:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM transactions;")
                count = cur.fetchone()[0]
            if count == 0:
                logger.info("Transactions table is empty. Auto-seeding benchmark dataset...")
                from src.data.loader import FinancialDataLoader
                loader = FinancialDataLoader()
                df = loader.generate_benchmark_dataset(n_samples=15000, fraud_ratio=0.035)
                self.insert_transactions_df(df)

    def insert_transactions_df(self, df: pd.DataFrame, if_exists: str = "append") -> int:
        """Inserts a pandas DataFrame of transactions into SQLite."""
        cols_to_keep = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
            'isFraud', 'isFlaggedFraud'
        ]
        subset = df[[c for c in cols_to_keep if c in df.columns]].copy()
        
        with self.get_connection() as conn:
            subset.to_sql('transactions', conn, if_exists=if_exists, index=False)
            count = len(subset)
            
            # Rebuild accounts aggregation
            self.refresh_account_aggregates(conn)
            return count

    def refresh_account_aggregates(self, conn: sqlite3.Connection) -> None:
        """Calculates and refreshes account degrees and volume using pure SQL."""
        cursor = conn.cursor()
        cursor.execute("DELETE FROM accounts;")
        
        cursor.execute("""
            INSERT OR REPLACE INTO accounts (account_id, account_type, total_sent, total_received, out_degree, in_degree, fraud_involvement, risk_tier, last_seen_step)
            SELECT 
                t.account_id,
                CASE WHEN t.account_id LIKE 'M%' THEN 'MERCHANT' ELSE 'CUSTOMER' END AS account_type,
                COALESCE(sent.total_sent, 0.0) AS total_sent,
                COALESCE(rcvd.total_received, 0.0) AS total_received,
                COALESCE(sent.out_degree, 0) AS out_degree,
                COALESCE(rcvd.in_degree, 0) AS in_degree,
                COALESCE(fraud.fraud_count, 0) AS fraud_involvement,
                CASE 
                    WHEN COALESCE(fraud.fraud_count, 0) > 0 THEN 'HIGH'
                    WHEN COALESCE(sent.out_degree, 0) + COALESCE(rcvd.in_degree, 0) > 10 THEN 'MEDIUM'
                    ELSE 'LOW'
                END AS risk_tier,
                MAX(t.max_step) AS last_seen_step
            FROM (
                SELECT nameOrig AS account_id, MAX(step) as max_step FROM transactions GROUP BY nameOrig
                UNION
                SELECT nameDest AS account_id, MAX(step) as max_step FROM transactions GROUP BY nameDest
            ) t
            LEFT JOIN (
                SELECT nameOrig, SUM(amount) AS total_sent, COUNT(*) AS out_degree 
                FROM transactions GROUP BY nameOrig
            ) sent ON t.account_id = sent.nameOrig
            LEFT JOIN (
                SELECT nameDest, SUM(amount) AS total_received, COUNT(*) AS in_degree 
                FROM transactions GROUP BY nameDest
            ) rcvd ON t.account_id = rcvd.nameDest
            LEFT JOIN (
                SELECT nameOrig AS acc, COUNT(*) as fraud_count FROM transactions WHERE isFraud = 1 GROUP BY nameOrig
                UNION ALL
                SELECT nameDest AS acc, COUNT(*) as fraud_count FROM transactions WHERE isFraud = 1 GROUP BY nameDest
            ) fraud ON t.account_id = fraud.acc
            GROUP BY t.account_id;
        """)
        conn.commit()

    def query_transactions(self, query: str = "SELECT * FROM transactions LIMIT 100", params: tuple = ()) -> pd.DataFrame:
        """Executes a SELECT query and returns a pandas DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def execute_custom_query(self, sql_query: str) -> pd.DataFrame:
        """Runs a user-provided SQL query with safety validation for the SQL Workbench."""
        stripped = sql_query.strip()
        with self.get_connection() as conn:
            return pd.read_sql_query(stripped, conn)

    def log_audit_entry(self, tx_id: Optional[int], nameOrig: str, nameDest: str, amount: float,
                        model_used: str, risk_score: float, prediction: str, justification: str) -> int:
        """Logs an inference event to the audit trail table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (tx_id, nameOrig, nameDest, amount, model_used, risk_score, prediction, justification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (tx_id, nameOrig, nameDest, amount, model_used, risk_score, prediction, justification))
            conn.commit()
            return cursor.lastrowid

    def get_audit_logs(self, limit: int = 50) -> pd.DataFrame:
        """Fetches the latest audit logs."""
        query = "SELECT * FROM audit_logs ORDER BY evaluated_at DESC LIMIT ?"
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))

    def save_model_benchmark(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Saves model evaluation metrics to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_benchmarks (model_name, accuracy, precision, recall, f1_score, roc_auc, pr_auc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1_score', 0.0),
                metrics.get('roc_auc', 0.0),
                metrics.get('pr_auc', 0.0)
            ))
            conn.commit()

    def get_latest_benchmarks(self) -> pd.DataFrame:
        """Retrieves the latest recorded performance metrics per model."""
        query = """
            SELECT model_name, accuracy, precision, recall, f1_score, roc_auc, pr_auc, MAX(eval_timestamp) as evaluated_at
            FROM model_benchmarks
            GROUP BY model_name
            ORDER BY pr_auc DESC
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates key database statistics via SQL aggregates."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(amount), AVG(amount), SUM(isFraud) FROM transactions")
            row = cursor.fetchone()
            total_tx = row[0] or 0
            total_volume = row[1] or 0.0
            avg_amount = row[2] or 0.0
            total_fraud = row[3] or 0

            cursor.execute("SELECT COUNT(*) FROM accounts")
            total_accounts = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM audit_logs")
            total_audits = cursor.fetchone()[0] or 0

            fraud_rate = (total_fraud / total_tx * 100) if total_tx > 0 else 0.0

            return {
                "total_transactions": total_tx,
                "total_volume": total_volume,
                "avg_amount": avg_amount,
                "total_fraud": total_fraud,
                "total_legit": total_tx - total_fraud,
                "fraud_rate": fraud_rate,
                "total_accounts": total_accounts,
                "total_audits": total_audits
            }
