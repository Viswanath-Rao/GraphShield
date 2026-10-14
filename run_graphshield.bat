@echo off
title GraphShield - Financial Fraud Intelligence Platform
cd /d "%~dp0"
color 0a

echo =======================================================================
echo          GRAPHSHIELD: FINANCIAL FRAUD DETECTION USING GNNs
echo =======================================================================
echo.
echo [1/3] Checking environment and pre-trained GNN model weights...
if not exist "models\metrics.json" (
    echo Models not found. Training baseline and GNN models...
    python -m src.training.trainer
) else (
    echo [OK] Pre-trained models and SQLite database verified!
)
echo.
echo [2/3] Opening browser at http://localhost:8501...
start "" "http://localhost:8501"

echo.
echo [3/3] Starting GraphShield Streamlit Command Center...
echo =======================================================================
echo Streamlit server active. Close this window or press Ctrl+C to stop.
echo =======================================================================
streamlit run app\streamlit_app.py --server.headless=false --server.port=8501
pause
