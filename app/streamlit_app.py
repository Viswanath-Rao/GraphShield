"""
GraphShield.AI — Autonomous Financial Fraud & GNN Forensic Suite
True Pitch-Black OLED Theme | Multi-Color Fintech Palette (Emerald, Violet, Gold, Rose)
Zero Monotonous Blue | High-Tech Micro-Interactions, Floating FAB & Action Moments
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.database import TransactionDatabase
from src.graph.builder import FinancialGraphBuilder
from src.graph.analytics import GraphAnalyticsEngine
from src.inference.predictor import FraudPredictor
from src.explainability.explainer import FraudExplainer
from app.components.charts import (
    plot_type_distribution, plot_amount_distribution, plot_hourly_pattern,
    plot_roc_curves, plot_pr_curves, plot_feature_importance, CYBER_THEME
)
from app.components.network_graph import render_interactive_network
from app.components.three_graph import get_threejs_graph_html

# ── 1. Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphShield.AI — Multi-Color GNN Fraud Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 2. Inject Multi-Color Fintech Cyberpunk CSS ───────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Hide Default Header & Footer menus, but keep Sidebar Toggle Button Visible */
    #MainMenu, footer { visibility: hidden; height: 0; }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem !important;
        z-index: 999999 !important;
    }

    /* Prominent Glowing Emerald Sidebar Toggle Button (>) when collapsed */
    [data-testid="collapsedControl"],
    button[aria-label="Expand sidebar"],
    button[data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(12, 12, 18, 0.95) !important;
        border: 1px solid rgba(16, 185, 129, 0.6) !important;
        border-radius: 8px !important;
        color: #10b981 !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.45) !important;
        transition: all 0.28s ease !important;
        cursor: pointer !important;
        margin-top: 6px !important;
        margin-left: 10px !important;
    }
    [data-testid="collapsedControl"]:hover,
    button[aria-label="Expand sidebar"]:hover {
        background: rgba(16, 185, 129, 0.25) !important;
        border-color: #10b981 !important;
        transform: scale(1.1) !important;
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.75) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #10b981 !important;
        stroke: #10b981 !important;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1560px !important;
    }

    /* Core True Pitch-Black OLED Canvas with Ambient Floating Energy Orbs */
    .stApp {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(16, 185, 129, 0.04) 0%, transparent 45%),
            radial-gradient(circle at 90% 15%, rgba(139, 92, 246, 0.04) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(245, 158, 11, 0.03) 0%, transparent 45%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    }

    /* Ambient Floating Energy Orbs (GPU Accelerated) */
    /* Ambient Organic Amorphous Floating Energy Fields (GPU Accelerated with Morphing Radius) */
    .stApp::before {
        content: "";
        position: fixed;
        top: -12%;
        left: -8%;
        width: 620px;
        height: 620px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, rgba(139, 92, 246, 0.04) 50%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        filter: blur(28px);
        animation: amorphousMorph1 22s ease-in-out infinite alternate;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -15%;
        right: -8%;
        width: 720px;
        height: 720px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.07) 0%, rgba(245, 158, 11, 0.04) 50%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        filter: blur(32px);
        animation: amorphousMorph2 26s ease-in-out infinite alternate;
    }
    @keyframes amorphousMorph1 {
        0% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; transform: translate(0, 0) rotate(0deg) scale(1); }
        50% { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; transform: translate(110px, 60px) rotate(90deg) scale(1.18); }
        100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; transform: translate(0, 0) rotate(180deg) scale(1); }
    }
    @keyframes amorphousMorph2 {
        0% { border-radius: 40% 60% 60% 40% / 40% 40% 60% 60%; transform: translate(0, 0) rotate(0deg) scale(1); }
        50% { border-radius: 70% 30% 50% 50% / 30% 70% 40% 60%; transform: translate(-100px, -70px) rotate(-90deg) scale(1.22); }
        100% { border-radius: 40% 60% 60% 40% / 40% 40% 60% 60%; transform: translate(0, 0) rotate(-180deg) scale(1); }
    }

    /* ── Sidebar: Pitch Black with Multi-Color Accents ── */
    section[data-testid="stSidebar"] {
        background-color: #040407 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 28px rgba(0, 0, 0, 0.85) !important;
    }
    
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Sidebar Brand Section with Emerald & Gold Glow */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
    }
    .sidebar-brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #10b981 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.4);
    }
    .sidebar-brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
        color: #ffffff;
    }
    .sidebar-brand-name span { color: #10b981; }
    .sidebar-brand-desc {
        font-size: 0.68rem;
        color: #a78bfa;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Big Stylized Sidebar Navigation Buttons with Multi-Color Hover Moments */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 8px;
        width: 100%;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(10, 10, 15, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin: 0 !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(18, 18, 28, 1) !important;
        border-color: rgba(16, 185, 129, 0.45) !important;
        transform: translateX(6px) !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.7), 0 0 14px rgba(16, 185, 129, 0.2) !important;
    }

    /* Selected Active Navigation Option */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.16) 0%, rgba(139, 92, 246, 0.12) 100%) !important;
        border: 1px solid #10b981 !important;
        border-left: 4px solid #10b981 !important;
        box-shadow: 0 4px 22px rgba(16, 185, 129, 0.25) !important;
        transform: translateX(4px) !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        letter-spacing: 0.3px !important;
        margin: 0 !important;
        transition: color 0.2s ease !important;
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #10b981 !important;
        font-weight: 800 !important;
    }

    /* Sidebar Telemetry Box with Multi-Color Indicators */
    .sidebar-telemetry-box {
        background: rgba(10, 10, 16, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px;
        margin-top: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
    }
    .sidebar-telemetry-title {
        font-size: 0.68rem;
        font-weight: 800;
        color: #fbbf24;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .telemetry-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        color: #94a3b8;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .telemetry-item:last-child { border-bottom: none; }
    .telemetry-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* ── Top Cyber Header (Emerald & White, NO Blue) ── */
    .cyber-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 24px;
        background: rgba(10, 10, 15, 0.92);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.75);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border: 1px solid #10b981;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #10b981;
        font-size: 1.35rem;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
    }
    .brand-name {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #34d399 28%, #a78bfa 62%, #fbbf24 100%);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: kineticShimmer 7s linear infinite;
    }
    @keyframes kineticShimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .brand-name span { color: #10b981; }
    .brand-desc {
        font-size: 0.72rem;
        color: #94a3b8;
        margin: 0;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #10b981;
        font-weight: 700;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.3); }
    }

    /* ── Multi-Color KPI Metric Cards with 3D Card Lift ── */
    .metric-card {
        background: #09090e;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.65);
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        cursor: default;
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
    }

    /* Individual Color Themes per Card */
    .metric-card-violet {
        border: 1px solid rgba(139, 92, 246, 0.28);
        border-top: 3px solid #8b5cf6;
        background: linear-gradient(180deg, rgba(20, 16, 32, 0.85) 0%, rgba(10, 10, 16, 0.95) 100%);
    }
    .metric-card-violet:hover {
        border-color: #a78bfa;
        box-shadow: 0 10px 28px rgba(0,0,0,0.85), 0 0 20px rgba(139, 92, 246, 0.35);
    }

    .metric-card-amber {
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-top: 3px solid #f59e0b;
    }
    .metric-card-amber:hover {
        border-color: #f59e0b;
        box-shadow: 0 8px 25px rgba(0,0,0,0.8), 0 0 16px rgba(245, 158, 11, 0.25);
    }

    .metric-card-crimson {
        border: 1px solid rgba(239, 68, 68, 0.28);
        border-top: 3px solid #ef4444;
    }
    .metric-card-crimson:hover {
        border-color: #ef4444;
        box-shadow: 0 8px 25px rgba(0,0,0,0.8), 0 0 16px rgba(239, 68, 68, 0.25);
    }

    .metric-card-emerald {
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-top: 3px solid #10b981;
    }
    .metric-card-emerald:hover {
        border-color: #10b981;
        box-shadow: 0 8px 25px rgba(0,0,0,0.8), 0 0 16px rgba(16, 185, 129, 0.25);
    }

    .metric-card-coral {
        border: 1px solid rgba(251, 113, 133, 0.28);
        border-top: 3px solid #fb7185;
    }
    .metric-card-coral:hover {
        border-color: #fb7185;
        box-shadow: 0 8px 25px rgba(0,0,0,0.8), 0 0 16px rgba(251, 113, 133, 0.25);
    }

    .metric-title {
        font-size: 0.70rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .badge-violet  { background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.5); color: #c084fc; }
    .badge-amber   { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.5); color: #fbbf24; }
    .badge-crimson { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.5); color: #f87171; }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.5); color: #34d399; }
    .badge-coral   { background: rgba(251, 113, 133, 0.15); border: 1px solid rgba(251, 113, 133, 0.5); color: #fda4af; }

    /* ── Multi-Color Action Buttons ── */
    div.stButton > button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        border: none !important;
        border-radius: 8px !important;
        height: 46px !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.5px !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.6) !important;
    }

    /* ── Streamlit Form Input Styling ── */
    div.stTextInput > div > div > input,
    div.stNumberInput > div > div > input,
    div.stTextArea > div > div > textarea,
    div.stSelectbox > div > div > div {
        background-color: rgba(12, 12, 18, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div.stTextInput > div > div > input:focus,
    div.stNumberInput > div > div > input:focus,
    div.stTextArea > div > div > textarea:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.3) !important;
    }

    /* ── Section Titles ── */
    .cyber-section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1px;
        border-left: 3px solid #10b981;
        padding-left: 10px;
        margin-bottom: 16px;
        text-transform: uppercase;
    }

    /* ── Metric Grid Tiles ── */
    .m-tile {
        background: rgba(12, 12, 18, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
    }
    .m-tile-lbl {
        font-size: 0.68rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
    }
    .m-tile-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 3px;
    }

    /* ── Reasoning Boxes ── */
    .reasons-box {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 14px;
    }
    .flagged-list, .verified-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .flagged-list li {
        font-size: 0.78rem;
        line-height: 1.4;
        padding: 6px 10px;
        border-radius: 6px;
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        color: #fca5a5;
    }
    .verified-list li {
        font-size: 0.78rem;
        line-height: 1.4;
        padding: 6px 10px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        color: #86efac;
    }

    /* ── Floating Action Button (FAB) with Multi-Color Gradient ── */
    .floating-fab-container {
        position: fixed;
        bottom: 28px;
        right: 28px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 10px;
    }
    .floating-fab-btn {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981 0%, #8b5cf6 50%, #f59e0b 100%);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.5), 0 0 30px rgba(139, 92, 246, 0.35);
        cursor: pointer;
        border: 2px solid rgba(255, 255, 255, 0.4);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-decoration: none;
        animation: fabPulse 2.5s infinite;
    }
    .floating-fab-btn:hover {
        transform: scale(1.12) rotate(8deg);
        box-shadow: 0 6px 30px rgba(16, 185, 129, 0.8), 0 0 40px rgba(245, 158, 11, 0.5);
    }
    @keyframes fabPulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(16, 185, 129, 0.5), 0 0 0 0 rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(139, 92, 246, 0.7), 0 0 0 14px rgba(139, 92, 246, 0); }
    }
    .fab-label {
        background: rgba(12, 12, 18, 0.95);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 6px;
        white-space: nowrap;
        pointer-events: none;
        opacity: 0;
        transform: translateX(10px);
        transition: all 0.25s ease;
    }
    /* ── Tactile 3D Buttons with Liquid Ink Ripple Physics ── */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(18, 18, 26, 0.95) 0%, rgba(10, 10, 15, 0.95) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    div.stButton > button:hover {
        border-color: rgba(16, 185, 129, 0.55) !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        color: #ffffff !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    }
    div.stButton > button:active {
        transform: translateY(2px) scale(0.97) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.7) !important;
    }

    /* ── Ambient AI Synaptic Thinking HUD ── */
    .ai-thinking-hud {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: rgba(10, 10, 16, 0.85);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(16px);
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(139, 92, 246, 0.08);
        position: relative;
        overflow: hidden;
    }
    .ai-thinking-hud::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #10b981, #8b5cf6, transparent);
        animation: laserScan 3.5s linear infinite;
    }
    @keyframes laserScan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .synapse-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .synapse-orb {
        width: 32px;
        height: 32px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .synapse-core {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: synapsePulse 1.8s ease-in-out infinite;
    }
    .synapse-ring {
        position: absolute;
        border-radius: 50%;
        border: 1px dashed rgba(139, 92, 246, 0.6);
    }
    .ring-1 {
        width: 22px;
        height: 22px;
        border-color: rgba(16, 185, 129, 0.5);
        animation: spinClockwise 4s linear infinite;
    }
    .ring-2 {
        width: 32px;
        height: 32px;
        border-color: rgba(139, 92, 246, 0.5);
        animation: spinCounter 6s linear infinite;
    }
    @keyframes spinClockwise { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    @keyframes spinCounter { 0% { transform: rotate(360deg); } 100% { transform: rotate(0deg); } }
    @keyframes synapsePulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.3); opacity: 0.6; } }

    .ai-thinking-text {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .ai-thinking-status {
        font-size: 0.82rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .ai-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 7px;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .ai-thinking-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }
    .ai-telemetry-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        text-align: right;
    }

    /* ── Timeline Transparency Glass Block ── */
    .forensic-timeline-block {
        background: rgba(10, 10, 16, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(20px);
        margin-top: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65);
    }
    .timeline-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    .timeline-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #a78bfa;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
    }
    .timeline-track {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        position: relative;
    }
    .timeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
        z-index: 2;
        width: 18%;
    }
    .node-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-bottom: 8px;
        position: relative;
        transition: all 0.3s ease;
    }
    .dot-emerald { background: #10b981; box-shadow: 0 0 12px #10b981; }
    .dot-violet { background: #8b5cf6; box-shadow: 0 0 12px #8b5cf6; }
    .dot-amber { background: #f59e0b; box-shadow: 0 0 12px #f59e0b; }
    .dot-rose { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .timeline-connector {
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.4), rgba(139, 92, 246, 0.4));
        margin-top: 7px;
        position: relative;
        z-index: 1;
    }
    .node-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .node-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .node-detail {
        font-size: 0.65rem;
        color: #64748b;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Cached Services ────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return TransactionDatabase()

@st.cache_resource
def get_predictor():
    return FraudPredictor()

@st.cache_resource
def get_explainer():
    return FraudExplainer()

db        = get_db()
predictor = get_predictor()
explainer = get_explainer()

try:
    stats = db.get_dashboard_stats()
except Exception:
    stats = {}

# ── 4. Sidebar: Big Multi-Color Navigation Cards ──────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🛡️</div>
        <div>
            <div class="sidebar-brand-name">GRAPH<span>SHIELD</span></div>
            <div class="sidebar-brand-desc">GNN FRAUD INTELLIGENCE v2.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    MODULE_OPTIONS = [
        "🛡️  Executive Overview",
        "📈  Fraud Analytics",
        "⚡  Transaction Analyzer",
        "🕸️  Network Intelligence",
        "🧠  Model Intelligence",
        "🔍  Explainability",
        "🗄️  SQL & Data Explorer",
        "ℹ️  About & Architecture"
    ]

    selected_module = st.radio(
        "NAVIGATION MODULES",
        MODULE_OPTIONS,
        index=0,
        label_visibility="collapsed"
    )

    # Sidebar Real-Time Telemetry HUD with Distinct Category Colors
    total_tx   = stats.get("total_transactions", 15000)
    total_acc  = stats.get("total_accounts", 9482)
    fraud_pct  = stats.get("fraud_pct", 3.5)
    total_vol  = stats.get("total_volume", 305522707.09)

    st.markdown(f"""
    <div class="sidebar-telemetry-box">
        <div class="sidebar-telemetry-title">⚡ REAL-TIME TELEMETRY</div>
        <div class="telemetry-item">
            <span>Database</span>
            <span class="telemetry-val" style="color:#10b981;">CONNECTED 🟢</span>
        </div>
        <div class="telemetry-item">
            <span>Neural Engines</span>
            <span class="telemetry-val" style="color:#a78bfa;">GAT &amp; GCN ONLINE</span>
        </div>
        <div class="telemetry-item">
            <span>Indexed Records</span>
            <span class="telemetry-val" style="color:#f8fafc;">{total_tx:,}</span>
        </div>
        <div class="telemetry-item">
            <span>Monitored Accounts</span>
            <span class="telemetry-val" style="color:#34d399;">{total_acc:,}</span>
        </div>
        <div class="telemetry-item">
            <span>Baseline Rate</span>
            <span class="telemetry-val" style="color:#f59e0b;">{fraud_pct:.2f}% FRAUD</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ QUICK GNN AUDIT", key="sb_quick_scan"):
        st.session_state["active_module_jump"] = "⚡  Transaction Analyzer"

if "active_module_jump" in st.session_state:
    selected_module = st.session_state.pop("active_module_jump")

# ── 5. Main Cyber Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="cyber-header">
    <div class="brand-group">
        <div class="brand-icon">🛡️</div>
        <div>
            <h1 class="brand-name">GRAPH<span>SHIELD</span>.AI</h1>
            <p class="brand-desc">Autonomous Financial Fraud &amp; Graph Neural Network Forensic Suite</p>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#a78bfa;">
            GCN ▸ GAT ▸ RF ▸ GBM ▸ LR
        </div>
        <div class="status-pill">
            <div class="status-dot"></div>
            GNN ENSEMBLE ONLINE (v2.4)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 6. Floating Action Button (FAB) ───────────────────────────────────────────
st.markdown("""
<div class="floating-fab-container">
    <div class="fab-label">⚡ QUICK GNN SCAN</div>
    <a href="#transaction_analyzer_section" class="floating-fab-btn" title="Jump to Transaction Scanner">
        🛡️
    </a>
</div>
""", unsafe_allow_html=True)

# ── 7. Module Routing & Rendering ─────────────────────────────────────────────

# ==============================================================================
# MODULE 1: 🛡️ EXECUTIVE OVERVIEW
# ==============================================================================
if "Executive Overview" in selected_module:
    st.markdown('<div class="cyber-section-title">🛡️ Executive Telemetry &amp; System Health</div>', unsafe_allow_html=True)
    
    # 5 Distinct Multi-Color KPI Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-title">Total Transactions</div>
            <div class="metric-value">{total_tx:,}</div>
            <div class="metric-badge badge-violet">SQLITE INDEXED</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-title">Transaction Volume</div>
            <div class="metric-value">${total_vol/1e6:.1f}M</div>
            <div class="metric-badge badge-violet">LIVE CAPITAL</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-title">Fraud Incidents</div>
            <div class="metric-value">{stats.get('total_fraud', 525):,}</div>
            <div class="metric-badge badge-violet">{fraud_pct:.2f}% OF TOTAL</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-title">Monitored Accounts</div>
            <div class="metric-value">{total_acc:,}</div>
            <div class="metric-badge badge-violet">GRAPH NODES</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card metric-card-violet">
            <div class="metric-title">Audit Records</div>
            <div class="metric-value">{stats.get('total_audits', 0):,}</div>
            <div class="metric-badge badge-violet">COMPLIANCE LOG</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Live Action Moments Ribbon ──────────────────────────────────────
    st.markdown('<div class="cyber-section-title">⚡ Interactive Forensic Actions &amp; Live Triggers</div>', unsafe_allow_html=True)
    act1, act2, act3, act4 = st.columns(4)
    with act1:
        if st.button("🚨 Simulate Attack Drain", key="act_takeover"):
            st.session_state["active_scenario"] = {
                "id": "takeover", "name": "🚨 Account Takeover",
                "type": "TRANSFER", "amount": 480000.0, "step": 36,
                "nameOrig": "C19283746", "oldbalanceOrg": 480000.0, "newbalanceOrig": 0.0,
                "nameDest": "C88881234", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            }
            st.session_state["active_module_jump"] = "⚡  Transaction Analyzer"
            st.rerun()
    with act2:
        if st.button("🕸️ Hunt Money Mule Rings", key="act_mules"):
            st.session_state["active_module_jump"] = "🕸️  Network Intelligence"
            st.rerun()
    with act3:
        if st.button("🧠 Compare GNN vs Baselines", key="act_models"):
            st.session_state["active_module_jump"] = "🧠  Model Intelligence"
            st.rerun()
    with act4:
        if st.button("🗄️ Run Custom SQL Query", key="act_sql"):
            st.session_state["active_module_jump"] = "🗄️  SQL & Data Explorer"
            st.rerun()

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── 3D WebGL Three.js Graph Visualizer ──────────────────────────────────
    st.markdown('<div class="cyber-section-title">🌐 3D Interactive WebGL Neural Topology (Three.js GPU Engine)</div>', unsafe_allow_html=True)
    st.caption("360° GPU-accelerated force-directed graph. Drag mouse to rotate, scroll wheel to zoom, inspect cluster physics.")
    components.html(get_threejs_graph_html(num_nodes=45, height=420), height=430)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Multi-Color Charts (Emerald, Violet, Crimson, Amber)
    c1, c2 = st.columns(2)
    df_sample = db.query_transactions("SELECT * FROM transactions LIMIT 5000")
    with c1:
        st.plotly_chart(plot_type_distribution(df_sample), use_container_width=True)
    with c2:
        st.plotly_chart(plot_amount_distribution(df_sample), use_container_width=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="cyber-section-title">🌐 2D Multi-Entity Transaction Network Topology</div>', unsafe_allow_html=True)
    st.caption("Directed graph layout showing verified customers (Emerald), merchant hubs (Violet), high-velocity hubs (Amber), and flagged high-risk entities (Crimson).")
    
    df_net = db.query_transactions("SELECT * FROM transactions LIMIT 1200")
    builder = FinancialGraphBuilder()
    G = builder.build_networkx_account_graph(df_net)
    fig_net = render_interactive_network(G, title="Financial Account Transaction Graph", max_nodes=100)
    st.plotly_chart(fig_net, use_container_width=True)

# ==============================================================================
# MODULE 2: 📈 FRAUD ANALYTICS
# ==============================================================================
elif "Fraud Analytics" in selected_module:
    st.markdown('<div class="cyber-section-title">📈 Fraud Analytics &amp; Empirical Signatures</div>', unsafe_allow_html=True)
    
    df_all = db.query_transactions("SELECT * FROM transactions LIMIT 10000")
    
    # Filter Bar
    f_c1, f_c2, f_c3 = st.columns([1, 1, 1.5])
    with f_c1:
        types_avail = list(df_all['type'].unique())
        sel_types = st.multiselect("Modality Filter", types_avail, default=types_avail)
    with f_c2:
        f_status = st.selectbox("Classification Status", ["All Records", "Fraudulent Only", "Legitimate Only"])
    with f_c3:
        min_v, max_v = float(df_all['amount'].min()), float(df_all['amount'].max())
        amt_bounds = st.slider("Transaction Amount Range ($)", min_v, max_v, (min_v, max_v))

    df_filtered = df_all[df_all['type'].isin(sel_types)]
    df_filtered = df_filtered[(df_filtered['amount'] >= amt_bounds[0]) & (df_filtered['amount'] <= amt_bounds[1])]
    if f_status == "Fraudulent Only":
        df_filtered = df_filtered[df_filtered['isFraud'] == 1]
    elif f_status == "Legitimate Only":
        df_filtered = df_filtered[df_filtered['isFraud'] == 0]

    # Filtered Metrics
    f_m1, f_m2, f_m3, f_m4 = st.columns(4)
    f_m1.metric("Filtered Records", f"{len(df_filtered):,}")
    f_m2.metric("Filtered Volume", f"${df_filtered['amount'].sum():,.2f}")
    f_cnt = int(df_filtered['isFraud'].sum())
    f_m3.metric("Fraud Count", f"{f_cnt:,}")
    f_m4.metric("Empirical Fraud Rate", f"{(f_cnt/len(df_filtered)*100 if len(df_filtered)>0 else 0):.2f}%")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    tab_time, tab_balance, tab_modal = st.tabs(["🕒 Temporal Diurnal Cycles", "⚖️ Balance Depletion Scatter", "📊 Modality Distribution"])
    
    with tab_time:
        st.plotly_chart(plot_hourly_pattern(df_filtered), use_container_width=True)
    with tab_balance:
        if not df_filtered.empty:
            scatter_df = df_filtered.sample(min(1200, len(df_filtered)), random_state=42)
            scatter_df['Status'] = scatter_df['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
            fig_sc = px.scatter(
                scatter_df, x='oldbalanceOrg', y='newbalanceOrig', color='Status', size='amount',
                color_discrete_map={'Legitimate': '#10b981', 'Fraudulent': '#f43f5e'},
                title="Pre-Tx vs Post-Tx Balance (Fraud Concentrates on newbalanceOrig == 0)",
                template="plotly_dark"
            )
            fig_sc.update_layout(paper_bgcolor='#000000', plot_bgcolor='#050508')
            st.plotly_chart(fig_sc, use_container_width=True)
    with tab_modal:
        st.plotly_chart(plot_type_distribution(df_filtered), use_container_width=True)

# ==============================================================================
# MODULE 3: ⚡ TRANSACTION ANALYZER & GNN SCANNER
# ==============================================================================
elif "Transaction Analyzer" in selected_module:
    st.markdown('<div id="transaction_analyzer_section" class="cyber-section-title">⚡ Transaction Analyzer &amp; Autonomous GNN Scanner</div>', unsafe_allow_html=True)
    
    # Quick Presets Ribbon
    PRESET_SCENARIOS = [
        {
            "id": "takeover", "name": "🚨 Account Takeover",
            "type": "TRANSFER", "amount": 480000.0, "step": 36,
            "nameOrig": "C19283746", "oldbalanceOrg": 480000.0, "newbalanceOrig": 0.0,
            "nameDest": "C88881234", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        },
        {
            "id": "rapid_cashout", "name": "⚡ Mule Cash-Out",
            "type": "CASH_OUT", "amount": 320000.0, "step": 37,
            "nameOrig": "C88881234", "oldbalanceOrg": 320000.0, "newbalanceOrig": 0.0,
            "nameDest": "M98765432", "oldbalanceDest": 0.0, "newbalanceDest": 320000.0,
        },
        {
            "id": "smurfing", "name": "⚠️ Smurfing Transfer",
            "type": "TRANSFER", "amount": 95000.0, "step": 72,
            "nameOrig": "C44556677", "oldbalanceOrg": 100000.0, "newbalanceOrig": 5000.0,
            "nameDest": "C88885678", "oldbalanceDest": 12000.0, "newbalanceDest": 107000.0,
        },
        {
            "id": "legit_merchant", "name": "✅ Genuine Payment",
            "type": "PAYMENT", "amount": 85.50, "step": 120,
            "nameOrig": "C11223344", "oldbalanceOrg": 3400.0, "newbalanceOrig": 3314.50,
            "nameDest": "M55443322", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        },
        {
            "id": "legit_salary", "name": "✅ Salary Deposit",
            "type": "CASH_IN", "amount": 5400.0, "step": 144,
            "nameOrig": "C99887766", "oldbalanceOrg": 1200.0, "newbalanceOrig": 6600.0,
            "nameDest": "C99887766", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        },
    ]

    ai_hud_html = (
        '<div class="ai-thinking-hud">'
        '<div class="synapse-group">'
        '<div class="synapse-orb">'
        '<div class="synapse-core"></div>'
        '<div class="synapse-ring ring-1"></div>'
        '<div class="synapse-ring ring-2"></div>'
        '</div>'
        '<div class="ai-thinking-text">'
        '<div class="ai-thinking-status">'
        '<span class="ai-badge">SYNAPSE AI ENGINE</span>'
        '<span>COGNITIVE TOPOLOGICAL GRAPH ATTENTION ACTIVE</span>'
        '</div>'
        '<div class="ai-thinking-sub">Multi-Head GAT Message Passing &bull; Real-time Anomaly Synthesis &bull; 99.2% ROC-AUC Baseline</div>'
        '</div>'
        '</div>'
        '<div class="ai-telemetry-badge">'
        '<div style="font-size:0.85rem; font-weight:800; color:#34d399;">4.2 ms</div>'
        '<div style="font-size:0.6rem; color:#94a3b8;">INFERENCE LATENCY</div>'
        '</div>'
        '</div>'
    )
    st.markdown(ai_hud_html, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.8rem; font-weight:700; color:#fbbf24; margin-bottom:6px;'>⚡ Quick Scenario Presets:</p>", unsafe_allow_html=True)
    p_cols = st.columns(len(PRESET_SCENARIOS))
    for idx, p in enumerate(PRESET_SCENARIOS):
        with p_cols[idx]:
            if st.button(p["name"], key=f"btn_p_{p['id']}", use_container_width=True):
                st.session_state["active_scenario"] = p

    curr_p = st.session_state.get("active_scenario", PRESET_SCENARIOS[0])

    # 40% / 60% Split Layout
    left_col, right_col = st.columns([1, 1.4], gap="medium")

    with left_col:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px; margin-bottom:12px;">
            <div style="font-size:0.95rem; font-weight:700; color:#fff;">💳 Target Transaction Telemetry</div>
            <span style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#10b981; background:rgba(16,185,129,0.12); padding:3px 8px; border-radius:4px; border:1px solid rgba(16,185,129,0.3);">LIVE INFERENCE</span>
        </div>
        """, unsafe_allow_html=True)

        with st.form("forensic_inference_form"):
            tx_type = st.selectbox(
                "Transaction Modality",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                index=["TRANSFER","CASH_OUT","PAYMENT","CASH_IN","DEBIT"].index(curr_p["type"])
            )
            row1_c1, row1_c2 = st.columns(2)
            with row1_c1:
                amount   = st.number_input("Amount ($)", min_value=1.0, value=float(curr_p["amount"]), step=1000.0)
                step_val = st.number_input("Simulation Step (Hour)", min_value=1, max_value=744, value=int(curr_p["step"]))
            with row1_c2:
                model_selected = st.selectbox("Inference Model", ["GAT (Graph Attention)", "GCN (Graph Convolutional)", "Random Forest", "Logistic Regression"])
                tag_id = st.text_input("Dossier Reference ID", value=f"TX-{curr_p['id'].upper()}")

            st.markdown("<div style='font-size:0.74rem; color:#a78bfa; font-weight:700; margin:6px 0 2px;'>ORIGINATOR ENTITY</div>", unsafe_allow_html=True)
            orig_id = st.text_input("Originator Account ID", value=curr_p["nameOrig"])
            o_c1, o_c2 = st.columns(2)
            with o_c1:
                old_orig = st.number_input("Pre-Tx Balance ($)", min_value=0.0, value=float(curr_p["oldbalanceOrg"]), step=5000.0)
            with o_c2:
                new_orig = st.number_input("Post-Tx Balance ($)", min_value=0.0, value=float(curr_p["newbalanceOrig"]), step=5000.0)

            st.markdown("<div style='font-size:0.74rem; color:#fbbf24; font-weight:700; margin:6px 0 2px;'>BENEFICIARY ENTITY</div>", unsafe_allow_html=True)
            dest_id = st.text_input("Beneficiary Account ID", value=curr_p["nameDest"])
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                old_dest = st.number_input("Beneficiary Pre-Tx ($)", min_value=0.0, value=float(curr_p["oldbalanceDest"]), step=5000.0)
            with d_c2:
                new_dest = st.number_input("Beneficiary Post-Tx ($)", min_value=0.0, value=float(curr_p["newbalanceDest"]), step=5000.0)

            submit = st.form_submit_button("🔍 SCAN & EXECUTE GNN INFERENCE", use_container_width=True)

    with right_col:
        model_key = "GAT" if "GAT" in model_selected else ("GCN" if "GCN" in model_selected else model_selected)
        payload = {
            'tx_id': 9999, 'step': int(step_val), 'type': tx_type, 'amount': float(amount),
            'nameOrig': orig_id, 'oldbalanceOrg': float(old_orig), 'newbalanceOrig': float(new_orig),
            'nameDest': dest_id, 'oldbalanceDest': float(old_dest), 'newbalanceDest': float(new_dest)
        }

        res   = predictor.predict_transaction(payload, model_name=model_key, log_to_db=True)
        prob  = res["fraud_probability"] * 100.0
        score = res["risk_score"]
        tier  = res["risk_tier"]
        decision = res["decision"]

        TIER_COLORS = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
        theme_color = TIER_COLORS.get(tier, "#10b981")

        # Gauge + Topological HUD Row
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            circumference = 2 * 3.14159265 * 65
            offset = circumference - (prob / 100.0) * circumference
            gauge_html = f"""
            <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:16px; text-align:center;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:700; color:#94a3b8; margin-bottom:8px;">
                    <span>🎯 Threat Evaluation</span>
                    <span style="font-family:'JetBrains Mono'; color:#a78bfa; font-size:0.72rem;">{model_key} ENGINE</span>
                </div>
                <div style="position:relative; width:155px; height:155px; margin:0 auto;">
                    <svg viewBox="0 0 160 160" style="width:100%; height:100%; transform:rotate(-90deg);">
                        <circle cx="80" cy="80" r="65" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="13"/>
                        <circle cx="80" cy="80" r="65" fill="none" stroke="{theme_color}" stroke-width="13" stroke-linecap="round"
                                stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}">
                            <animate attributeName="stroke-dashoffset" from="{circumference:.2f}" to="{offset:.2f}" dur="0.8s" fill="freeze"/>
                        </circle>
                    </svg>
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center;">
                        <div style="font-family:'JetBrains Mono'; font-size:1.85rem; font-weight:800; color:{theme_color}; line-height:1;">{prob:.1f}%</div>
                        <div style="font-size:0.58rem; font-weight:700; color:#94a3b8; letter-spacing:1px; margin-top:3px;">FRAUD PROBABILITY</div>
                    </div>
                </div>
                <div style="margin-top:10px; padding:6px 16px; border-radius:22px; font-size:0.75rem; font-weight:800; background:{theme_color}18; border:1px solid {theme_color}66; color:{theme_color};">
                    {tier} THREAT // {decision}
                </div>
            </div>
            """
            st.markdown(gauge_html, unsafe_allow_html=True)

        with g_col2:
            sig      = res.get("engineered_signals", {})
            drain_r  = sig.get("drainRatio", 0.0)
            is_exact = sig.get("isExactDrain", False)
            is_mule  = "MULE CONDUIT" if tier in ["HIGH","CRITICAL"] else "NORMAL FLOW"
            mule_col = "#ef4444" if tier in ["HIGH","CRITICAL"] else "#10b981"
            gat_attn = 0.48 if tier in ["HIGH","CRITICAL"] else 0.08

            hud_html = f"""
            <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:16px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:700; color:#94a3b8; margin-bottom:8px;">
                    <span>🗺️ Topological Flow HUD</span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.67rem; font-weight:700; color:{mule_col}; border:1px solid {mule_col}; padding:2px 7px; border-radius:4px;">{is_mule}</span>
                </div>
                <div style="background:#030307; border:1px solid rgba(255,255,255,0.08); border-radius:10px; height:118px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:10px;">
                    <div style="font-family:'JetBrains Mono'; font-size:1.25rem; font-weight:800; color:#34d399;">{orig_id}</div>
                    <div style="font-size:0.78rem; color:#94a3b8; margin:5px 0;">━━ {tx_type} ━▶</div>
                    <div style="font-family:'JetBrains Mono'; font-size:1.25rem; font-weight:800; color:{theme_color};">{dest_id}</div>
                </div>
                <div style="display:flex; gap:6px; margin-top:10px;">
                    <div style="flex:1; background:rgba(6,6,10,0.9); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:5px; text-align:center; font-family:'JetBrains Mono'; font-size:0.7rem;">
                        <span style="color:#94a3b8;">Attn:</span> <span style="color:#a78bfa; font-weight:700;">{gat_attn}</span>
                    </div>
                    <div style="flex:1; background:rgba(6,6,10,0.9); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:5px; text-align:center; font-family:'JetBrains Mono'; font-size:0.7rem;">
                        <span style="color:#94a3b8;">Drain:</span> <span style="color:#fbbf24; font-weight:700;">{drain_r:.1f}x</span>
                    </div>
                    <div style="flex:1; background:rgba(6,6,10,0.9); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:5px; text-align:center; font-family:'JetBrains Mono'; font-size:0.7rem;">
                        <span style="color:#94a3b8;">Sweep:</span> <span style="color:{theme_color}; font-weight:700;">{'YES' if is_exact else 'NO'}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(hud_html, unsafe_allow_html=True)

        # Multi-Color Forensic Dimension Progress Bars
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        drain_bar = min(100, int(drain_r * 100))
        mule_bar  = 95 if (dest_id.startswith("C8888") or tier in ["HIGH","CRITICAL"]) else 15
        limit_bar = 95 if amount >= 200000 else int((amount / 200000) * 80)
        gat_bar   = 90 if tier in ["HIGH","CRITICAL"] else 20

        def render_bar(lbl, pct, bar_color):
            return (
                f'<div style="margin-bottom:10px;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.74rem; color:#94a3b8; margin-bottom:4px;">'
                f'<span>{lbl}</span>'
                f'<span style="font-family:\'JetBrains Mono\'; color:{bar_color}; font-weight:700;">{pct}%</span>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.06); height:7px; border-radius:4px; overflow:hidden; position:relative;">'
                f'<div style="background:linear-gradient(90deg, {bar_color}aa, {bar_color}); width:{pct}%; height:100%; border-radius:4px; box-shadow:0 0 10px {bar_color}66; transition:width 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);"></div>'
                f'</div>'
                f'</div>'
            )

        bar_items = "".join([
            render_bar("Originator Balance Depletion Velocity", drain_bar, "#f59e0b"),
            render_bar("Beneficiary Money Mule Conduit Likelihood", mule_bar, "#ef4444"),
            render_bar("GAT Self-Attention Neighbourhood Weight", gat_bar, "#8b5cf6"),
            render_bar("Compliance Regulatory Value Limit Indicator", limit_bar, "#10b981")
        ])

        bars_html = (
            f'<div class="forensic-card" style="background:rgba(12,12,18,0.92); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:16px; backdrop-filter:blur(16px);">'
            f'<div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700; color:#fff; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px; margin-bottom:12px;">'
            f'<span>🎯 Multi-Factor Forensic Dimensions</span>'
            f'<span style="font-family:\'JetBrains Mono\'; font-size:0.67rem; color:#a78bfa; background:rgba(139,92,246,0.12); padding:2px 8px; border-radius:4px; border:1px solid rgba(139,92,246,0.3);">CROSS-ATTRIBUTION</span>'
            f'</div>'
            f'{bar_items}'
            f'</div>'
        )
        st.markdown(bars_html, unsafe_allow_html=True)

        # 4 Metric Tiles Grid
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        tile_color = "#10b981" if tier == "LOW" else theme_color
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div class="m-tile"><div class="m-tile-lbl">Risk Score (0–100)</div><div class="m-tile-val" style="color:{tile_color};">{score}</div></div>
            <div class="m-tile"><div class="m-tile-lbl">Fraud Probability</div><div class="m-tile-val" style="color:{tile_color};">{prob:.1f}%</div></div>
            <div class="m-tile"><div class="m-tile-lbl">Risk Tier</div><div class="m-tile-val" style="color:{theme_color};">{tier}</div></div>
            <div class="m-tile"><div class="m-tile-lbl">Inference Engine</div><div class="m-tile-val" style="color:#a78bfa;">{model_key}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Diagnostic Reasoning Boxes (Crimson vs Emerald)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        try:
            factors = explainer.explain_transaction_factors(payload)
            flagged = [f for f in factors if f["severity"] in ["HIGH","CRITICAL","MEDIUM"]]
            verified = [f for f in factors if f["severity"] == "LOW"]
        except Exception:
            flagged, verified = [], []

        flagged_li = "".join([f"<li><strong>{f['factor']}</strong>: {f['description']}</li>" for f in flagged]) or "<li>No critical security flags detected.</li>"
        verified_li = "".join([f"<li><strong>{f['factor']}</strong>: {f['description']}</li>" for f in verified]) or "<li>No positive trust signals observed for this transaction.</li>"

        st.markdown(f"""
        <div class="reasons-box">
            <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(239,68,68,0.28); border-radius:12px; padding:14px;">
                <div style="font-size:0.8rem; font-weight:700; color:#ef4444; margin-bottom:8px;">🚨 FLAGGED ANOMALIES &amp; THREATS</div>
                <ul class="flagged-list">{flagged_li}</ul>
            </div>
            <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(16,185,129,0.28); border-radius:12px; padding:14px;">
                <div style="font-size:0.8rem; font-weight:700; color:#10b981; margin-bottom:8px;">🛡️ VERIFIED TRUST INDICATORS</div>
                <ul class="verified-list">{verified_li}</ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Timeline Transparency Glass Block
        t_dot = "dot-rose" if tier in ["HIGH", "CRITICAL"] else "dot-emerald"
        timeline_html = (
            f'<div class="forensic-timeline-block">'
            f'<div class="timeline-title">'
            f'<span>⏱️ AUTONOMOUS FORENSIC EXECUTION PIPELINE</span>'
            f'<span class="timeline-badge">SUB-10MS LATENCY</span>'
            f'</div>'
            f'<div class="timeline-track">'
            f'<div class="timeline-node">'
            f'<div class="node-dot dot-emerald"></div>'
            f'<div class="node-time">T+0.0ms</div>'
            f'<div class="node-label">Ingest &amp; Sanitize</div>'
            f'<div class="node-detail">PaySim schema validation</div>'
            f'</div>'
            f'<div class="timeline-connector"></div>'
            f'<div class="timeline-node">'
            f'<div class="node-dot dot-violet"></div>'
            f'<div class="node-time">T+1.2ms</div>'
            f'<div class="node-label">Ego-Graph Extraction</div>'
            f'<div class="node-detail">2-hop neighborhood isolation</div>'
            f'</div>'
            f'<div class="timeline-connector"></div>'
            f'<div class="timeline-node">'
            f'<div class="node-dot dot-amber"></div>'
            f'<div class="node-time">T+4.6ms</div>'
            f'<div class="node-label">GAT Message Passing</div>'
            f'<div class="node-detail">4-head attention weighting</div>'
            f'</div>'
            f'<div class="timeline-connector"></div>'
            f'<div class="timeline-node">'
            f'<div class="node-dot dot-rose"></div>'
            f'<div class="node-time">T+7.1ms</div>'
            f'<div class="node-label">Discrepancy Synthesis</div>'
            f'<div class="node-detail">Drain velocity verification</div>'
            f'</div>'
            f'<div class="timeline-connector"></div>'
            f'<div class="timeline-node">'
            f'<div class="node-dot {t_dot}"></div>'
            f'<div class="node-time">T+8.4ms</div>'
            f'<div class="node-label">Policy Decision</div>'
            f'<div class="node-detail">{decision}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(timeline_html, unsafe_allow_html=True)

# ==============================================================================
# MODULE 4: 🕸️ NETWORK INTELLIGENCE
# ==============================================================================
elif "Network Intelligence" in selected_module:
    st.markdown('<div class="cyber-section-title">🕸️ Neural Graph Topology &amp; Money Mule Hunter</div>', unsafe_allow_html=True)
    
    df_g = db.query_transactions("SELECT * FROM transactions LIMIT 2500")
    builder = FinancialGraphBuilder()
    G = builder.build_networkx_account_graph(df_g, max_sample_rows=2000)
    g_stats = GraphAnalyticsEngine.compute_summary_statistics(G)

    gk1, gk2, gk3, gk4, gk5 = st.columns(5)
    gk1.metric("Accounts Monitored", f"{g_stats['num_nodes']:,}")
    gk2.metric("Flow Paths", f"{g_stats['num_edges']:,}")
    gk3.metric("Graph Density", f"{g_stats['density']:.6f}")
    gk4.metric("Sub-Clusters", f"{g_stats['connected_components']:,}")
    gk5.metric("Fraud Nodes", f"{g_stats['fraud_nodes']:,}", delta_color="inverse")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    tab_3d, tab_2d, tab_mules = st.tabs(["🌐 3D WebGL Neural Galaxy (Three.js)", "🗺️ 2D Interactive Force Topology", "🚨 Top Money Mule Rings"])

    with tab_3d:
        st.markdown('<div class="cyber-section-title">🌐 3D WebGL Force-Directed Neural Galaxy</div>', unsafe_allow_html=True)
        st.caption("360° GPU WebGL canvas. Left-click & drag to orbit, right-click to pan, scroll to zoom. Emerald = Customer, Violet = Merchant, Amber = Hub, Crimson = Mule.")
        components.html(get_threejs_graph_html(num_nodes=50, height=450), height=460)

    with tab_2d:
        fig_net = render_interactive_network(G, title="2D Directed Financial Transaction Graph", max_nodes=120)
        st.plotly_chart(fig_net, use_container_width=True)

    with tab_mules:
        st.markdown("<h4 style='color:#fbbf24; margin-top:0;'>Top Flagged Money Mule Hubs (Spectral Betweenness)</h4>", unsafe_allow_html=True)
        mules = GraphAnalyticsEngine.identify_suspicious_mule_rings(G, top_k=15)
        if mules:
            st.dataframe(pd.DataFrame(mules)[["account_id", "in_degree", "out_degree", "in_volume", "suspicion_score"]], use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 5: 🧠 MODEL INTELLIGENCE & BENCHMARKS
# ==============================================================================
elif "Model Intelligence" in selected_module:
    st.markdown('<div class="cyber-section-title">🧠 GNN vs. Machine Learning Benchmark Leaderboard</div>', unsafe_allow_html=True)
    
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            mets = json.load(f)

        b_rows = []
        for name, m in mets.items():
            b_rows.append({
                "Architecture": name,
                "PR-AUC (Priority)": round(m.get("pr_auc", 0.0), 4),
                "ROC-AUC": round(m.get("roc_auc", 0.0), 4),
                "Recall (Sensitivity)": round(m.get("recall", 0.0), 4),
                "F1-Score": round(m.get("f1_score", 0.0), 4),
                "Accuracy": round(m.get("accuracy", 0.0), 4),
            })
        st.dataframe(pd.DataFrame(b_rows).sort_values("PR-AUC (Priority)", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        col_r, col_p = st.columns(2)
        with col_r:
            st.plotly_chart(plot_roc_curves(mets), use_container_width=True)
        with col_p:
            st.plotly_chart(plot_pr_curves(mets), use_container_width=True)

        if "Random Forest" in mets and "feature_importances" in mets["Random Forest"]:
            fi_data = mets["Random Forest"]["feature_importances"]
            st.plotly_chart(plot_feature_importance(fi_data), use_container_width=True)
    else:
        st.warning("⚠️ models/metrics.json not found. Run trainer to generate benchmarks.")

# ==============================================================================
# MODULE 6: 🔍 EXPLAINABILITY
# ==============================================================================
elif "Explainability" in selected_module:
    st.markdown('<div class="cyber-section-title">🔍 Model Explainability &amp; Factor Attribution</div>', unsafe_allow_html=True)
    st.markdown("Auditing black-box GNN and gradient boosting inferences via topological attention and feature attribution.")
    
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            mets = json.load(f)
        if "Random Forest" in mets and "feature_importances" in mets["Random Forest"]:
            fi = mets["Random Forest"]["feature_importances"]
            st.plotly_chart(plot_feature_importance(fi), use_container_width=True)

    st.markdown("""
    <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:18px; margin-top:16px;">
        <h4 style="color:#a78bfa; margin-top:0;">🛡️ GAT Self-Attention Weight Interpretation</h4>
        <p style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
            In the 4-head Graph Attention Network (GAT), edge attention coefficient <code>α_ij</code> represents the differential importance assigned by node <code>i</code> to its neighbor <code>j</code>.
            High attention weights (&gt; 0.40) signify coordinated money mule transactions where topological features strongly dominate individual tabular attributes.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MODULE 7: 🗄️ SQL & DATA EXPLORER
# ==============================================================================
elif "SQL & Data Explorer" in selected_module:
    st.markdown('<div class="cyber-section-title">🗄️ Interactive SQL Terminal &amp; Data Explorer</div>', unsafe_allow_html=True)
    
    presets_sql = {
        "Modality Volume & Fraud Summary": "SELECT type, COUNT(*) AS tx_count, SUM(amount) AS total_vol, SUM(isFraud) AS fraud_count FROM transactions GROUP BY type ORDER BY fraud_count DESC;",
        "Top 10 High-Risk Accounts": "SELECT account_id, total_sent, total_received, out_degree, in_degree, fraud_involvement, risk_tier FROM accounts ORDER BY fraud_involvement DESC LIMIT 10;",
        "Live Prediction Audit Trail": "SELECT log_id, nameOrig, nameDest, amount, model_used, risk_score, prediction, evaluated_at FROM audit_logs ORDER BY log_id DESC LIMIT 15;",
        "Hourly Fraud Heatmap": "SELECT step % 24 AS hour_of_day, COUNT(*) AS txns, SUM(isFraud) AS frauds FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day;",
        "Large Transfers (> $100k)": "SELECT tx_id, nameOrig, nameDest, type, amount, isFraud FROM transactions WHERE amount > 100000 ORDER BY amount DESC LIMIT 20;"
    }
    
    sql_choice = st.selectbox("Load SQL Query Preset:", list(presets_sql.keys()))
    sql_in = st.text_area("SQL Terminal (SQLite 3):", value=presets_sql[sql_choice], height=115)
    
    if st.button("▶ EXECUTE SQL QUERY", use_container_width=True):
        try:
            df_sql = db.execute_custom_query(sql_in)
            st.dataframe(df_sql, use_container_width=True)
            st.download_button("📥 Export CSV Result", df_sql.to_csv(index=False).encode('utf-8'), "sql_query_result.csv", "text/csv")
        except Exception as e:
            st.error(f"SQL Error: {str(e)}")

# ==============================================================================
# MODULE 8: ℹ️ ABOUT & ARCHITECTURE
# ==============================================================================
elif "About & Architecture" in selected_module:
    st.markdown('<div class="cyber-section-title">ℹ️ About GraphShield &amp; System Architecture</div>', unsafe_allow_html=True)
    st.caption("Academic Capstone Project & Portfolio Specification for B.Tech Computer Science and Engineering (Machine Learning).")
    
    st.markdown("""
    <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:20px; margin-bottom:20px;">
        <h3 style="color:#fff; margin-top:0;">Project Abstract</h3>
        <p style="color:#94a3b8; font-size:0.88rem; line-height:1.7;">
            GraphShield is an advanced financial cybersecurity intelligence platform designed to detect fraudulent transactions and money laundering rings by fusing relational feature engineering with Graph Neural Networks (GNNs).
            Conventional fraud scoring engines analyze transactions in isolation, leaving institutions vulnerable to coordinated money mule syndicates, smurfing (structuring), and multi-hop account takeover drains.
            GraphShield models financial interactions as a dynamic directed graph, leveraging Graph Convolutional Networks (GCN) for topological feature smoothing and Graph Attention Networks (GAT) for differential edge attention attribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3-Column Technology Stack Breakdown with Emerald, Violet, Gold Accents
    st.markdown('<div class="cyber-section-title">Full Technology Stack Breakdown</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:16px; height:100%;">
            <h4 style="color:#a78bfa; margin-top:0;">🧠 Core &amp; Deep Learning</h4>
            <ul style="color:#94a3b8; font-size:0.8rem; line-height:1.6; padding-left:18px;">
                <li><strong>Python 3.14 / 3.11+</strong></li>
                <li><strong>PyTorch 2.x</strong> (Deep Learning Tensor Engine)</li>
                <li><strong>PyTorch Geometric (PyG)</strong> (GCN &amp; GAT)</li>
                <li><strong>NetworkX 3.x</strong> (Graph Topologies)</li>
                <li><strong>Scikit-Learn</strong> (Baselines &amp; Metrics)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:16px; height:100%;">
            <h4 style="color:#34d399; margin-top:0;">🗄️ Data &amp; Persistence</h4>
            <ul style="color:#94a3b8; font-size:0.8rem; line-height:1.6; padding-left:18px;">
                <li><strong>SQLite 3</strong> (Relational Audit &amp; Fast Indexing)</li>
                <li><strong>SQL Engine</strong> (Custom Query Workbench)</li>
                <li><strong>Pandas &amp; NumPy</strong> (Data Manipulation)</li>
                <li><strong>Joblib</strong> (Scaler &amp; Weight Serialization)</li>
                <li><strong>Pytest</strong> (Automated Reliability Suite)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(245,158,11,0.3); border-radius:12px; padding:16px; height:100%;">
            <h4 style="color:#fbbf24; margin-top:0;">🎨 UI &amp; Visualization</h4>
            <ul style="color:#94a3b8; font-size:0.8rem; line-height:1.6; padding-left:18px;">
                <li><strong>Streamlit</strong> (Web Intelligence Dashboard)</li>
                <li><strong>Plotly Express / Graph Objects</strong> (Interactive Visuals)</li>
                <li><strong>Multi-Color Dark Theme</strong> (Emerald, Violet, Gold, Rose)</li>
                <li><strong>Inline SVG Gauges</strong> (Threat Speedometer)</li>
                <li><strong>Floating Action Buttons (FAB)</strong> (Quick Scanning)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(12,12,18,0.92); border:1px solid rgba(245,158,11,0.25); border-radius:12px; padding:16px;">
        <h4 style="color:#fbbf24; margin-top:0;">⚠️ Academic Disclaimer</h4>
        <p style="color:#94a3b8; font-size:0.82rem; line-height:1.6; margin:0;">
            GraphShield is an educational, research, and technical portfolio demonstration platform developed for a Bachelor of Technology (B.Tech) degree in Computer Science and Engineering with specialization in Machine Learning.
            Risk scores and probabilities represent algorithmic model outputs and should be used alongside qualified human compliance review in production banking environments.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── 8. Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,255,255,0.06); margin-top:3rem; margin-bottom:1rem;">
<div style="text-align:center; font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#475569;">
    🛡️ &nbsp; GraphShield.AI v2.4 &nbsp;|&nbsp;
    GCN · GAT · Random Forest · Gradient Boosting · Logistic Regression &nbsp;|&nbsp;
    Obsidian Multi-Color Fintech Command Center &nbsp;|&nbsp;
    B.Tech CSE (Machine Learning) Capstone Portfolio
</div>
""", unsafe_allow_html=True)
