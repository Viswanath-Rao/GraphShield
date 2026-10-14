-- =======================================================================
-- GraphShield Relational Database Schema & Analytical Queries
-- High-Performance SQLite 3 DDL for Transaction Forensics
-- =======================================================================

-- 1. Transactions Ledger Table
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

-- 2. Aggregated Account Profiles Table
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

-- 3. Live Compliance Audit Logs
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

-- 4. Benchmark Evaluation Metric Storage
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

-- High-Speed Covering Indexes
CREATE INDEX IF NOT EXISTS idx_tx_orig ON transactions(nameOrig);
CREATE INDEX IF NOT EXISTS idx_tx_dest ON transactions(nameDest);
CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_tx_fraud ON transactions(isFraud);
CREATE INDEX IF NOT EXISTS idx_tx_step ON transactions(step);
