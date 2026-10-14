"""
GraphShield UI Components - High-Tech Multi-Color Fintech & Cyber Charts
Sleek pitch-black canvas with rich multi-color palette (Emerald, Violet, Amber, Coral, Rose).
No monotonous blue/cyan!
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Neutral ultra-clean dark theme - NO blue tint!
CYBER_THEME = dict(
    paper_bgcolor='#000000',
    plot_bgcolor='#050508',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#f1f5f9', size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(
        gridcolor='rgba(255, 255, 255, 0.05)',
        zerolinecolor='rgba(255, 255, 255, 0.08)',
        tickfont=dict(family='JetBrains Mono', color='#94a3b8')
    ),
    yaxis=dict(
        gridcolor='rgba(255, 255, 255, 0.05)',
        zerolinecolor='rgba(255, 255, 255, 0.08)',
        tickfont=dict(family='JetBrains Mono', color='#94a3b8')
    ),
    legend=dict(
        bgcolor='rgba(12, 12, 18, 0.88)',
        bordercolor='rgba(255, 255, 255, 0.1)',
        borderwidth=1,
        font=dict(family='Plus Jakarta Sans', color='#f8fafc')
    )
)

def plot_type_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots transaction distribution by modality and classification.
    Legitimate uses rich Emerald Mint (#10b981), Fraud uses Vivid Crimson Rose (#f43f5e).
    """
    grouped = df.groupby(['type', 'isFraud']).size().reset_index(name='count')
    grouped['Status'] = grouped['isFraud'].map({0: 'Legitimate Flow', 1: 'Fraudulent Drain'})
    
    fig = px.bar(
        grouped,
        x='type',
        y='count',
        color='Status',
        barmode='group',
        color_discrete_map={
            'Legitimate Flow': '#10b981',    # Vivid Emerald Mint
            'Fraudulent Drain': '#f43f5e'    # Vivid Crimson Rose
        },
        title="Transaction Modality Volume (Emerald: Legit | Rose: Fraud)",
        labels={'count': 'Transaction Count', 'type': 'Transaction Modality'}
    )
    fig.update_layout(**CYBER_THEME)
    fig.update_traces(marker_line_width=1, marker_line_color='rgba(255, 255, 255, 0.15)')
    return fig

def plot_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots log-transformed transaction amount distributions.
    Legitimate uses Royal Violet (#8b5cf6), Fraud uses Vivid Ruby Crimson (#ef4444).
    """
    fig = go.Figure()
    
    legit_amounts = np.log10(df[df['isFraud'] == 0]['amount'] + 1)
    fraud_amounts = np.log10(df[df['isFraud'] == 1]['amount'] + 1)

    fig.add_trace(go.Histogram(
        x=legit_amounts,
        name='Legitimate Capital',
        marker=dict(color='#8b5cf6', line=dict(color='#a78bfa', width=1)),  # Royal Violet
        opacity=0.65,
        nbinsx=40
    ))
    fig.add_trace(go.Histogram(
        x=fraud_amounts,
        name='Fraudulent Drain',
        marker=dict(color='#ef4444', line=dict(color='#f87171', width=1)),  # Crimson Red
        opacity=0.85,
        nbinsx=40
    ))

    fig.update_layout(
        **CYBER_THEME,
        barmode='overlay',
        title="Capital Density Distribution (Violet: Legit | Crimson: Fraud)",
        xaxis_title="Log10(Amount in USD)",
        yaxis_title="Observed Frequency"
    )
    return fig

def plot_hourly_pattern(df: pd.DataFrame) -> go.Figure:
    """
    Plots transaction diurnal cycle with Amber/Gold for volume and Rose for fraud.
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy['step'] % 24
    hourly = df_copy.groupby(['hour', 'isFraud']).size().reset_index(name='count')
    hourly['Status'] = hourly['isFraud'].map({0: 'Legitimate Volume', 1: 'Fraud Spike'})

    fig = px.line(
        hourly,
        x='hour',
        y='count',
        color='Status',
        color_discrete_map={
            'Legitimate Volume': '#f59e0b',   # Warm Amber / Gold
            'Fraud Spike': '#f43f5e'          # Vivid Rose
        },
        markers=True,
        title="24-Hour Velocity Curve (Amber: Total Velocity | Rose: Fraud Spikes)",
        labels={'hour': 'Diurnal Hour (0-23)', 'count': 'Transaction Volume'}
    )
    fig.update_layout(**CYBER_THEME)
    return fig

def plot_confusion_matrix(cm_dict: Dict[str, int], model_name: str) -> go.Figure:
    """Renders a styled confusion matrix heatmap with Emerald & Crimson palette."""
    z = [
        [cm_dict.get('true_negative', 0), cm_dict.get('false_positive', 0)],
        [cm_dict.get('false_negative', 0), cm_dict.get('true_positive', 0)]
    ]
    x = ['Predicted Legit', 'Predicted Fraud']
    y = ['Actual Legit', 'Actual Fraud']

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=f"<b>{val:,}</b>",
                showarrow=False,
                font=dict(color='#ffffff', size=18, family='JetBrains Mono')
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, '#09090e'], [0.5, '#7c3aed'], [1, '#10b981']],
        showscale=False
    ))
    fig.update_layout(
        **CYBER_THEME,
        title=f"Diagnostic Confusion Matrix: {model_name}",
        annotations=annotations,
        xaxis=dict(side='top')
    )
    return fig

def plot_roc_curves(metrics_dict: Dict[str, Any]) -> go.Figure:
    """Plots multi-model ROC curve comparison with distinct multi-color traces."""
    fig = go.Figure()
    # Vibrant distinctive palette: Emerald, Amber, Violet, Rose, Sky
    colors = ['#10b981', '#f59e0b', '#8b5cf6', '#f43f5e', '#38bdf8']
    
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        fpr = metrics.get('fpr', [0, 1])
        tpr = metrics.get('tpr', [0, 1])
        auc_val = metrics.get('roc_auc', 0.5)
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{name} (AUC = {auc_val:.4f})",
            line=dict(color=color, width=2.5)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.5)',
        line=dict(color='rgba(255, 255, 255, 0.25)', width=1.5, dash='dash')
    ))

    theme = {k: v for k, v in CYBER_THEME.items() if k != 'legend'}
    fig.update_layout(
        **theme,
        title="Multi-Model ROC Diagnostic Benchmark",
        xaxis_title="False Positive Rate (Fallout)",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(
            x=0.55, y=0.08,
            bgcolor='rgba(12, 12, 18, 0.88)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1,
            font=dict(family='Plus Jakarta Sans', color='#f8fafc')
        )
    )
    return fig

def plot_pr_curves(metrics_dict: Dict[str, Any]) -> go.Figure:
    """Plots Precision-Recall curve with distinct multi-color traces."""
    fig = go.Figure()
    colors = ['#10b981', '#f59e0b', '#8b5cf6', '#f43f5e', '#38bdf8']

    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        rec = metrics.get('recall_curve', [0, 1])
        prec = metrics.get('precision_curve', [1, 0])
        pr_auc = metrics.get('pr_auc', 0.0)
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=rec, y=prec,
            mode='lines',
            name=f"{name} (PR-AUC = {pr_auc:.4f})",
            line=dict(color=color, width=2.5)
        ))

    theme = {k: v for k, v in CYBER_THEME.items() if k != 'legend'}
    fig.update_layout(
        **theme,
        title="Precision-Recall Curve (Imbalanced Benchmark)",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        legend=dict(
            x=0.55, y=0.08,
            bgcolor='rgba(12, 12, 18, 0.88)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1,
            font=dict(family='Plus Jakarta Sans', color='#f8fafc')
        )
    )
    return fig

def plot_feature_importance(fi_dict: Dict[str, float]) -> go.Figure:
    """Plots top feature importance with warm Amber-to-Violet gradient."""
    sorted_items = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    feats = [item[0] for item in sorted_items][::-1]
    scores = [item[1] for item in sorted_items][::-1]

    fig = go.Figure(go.Bar(
        x=scores,
        y=feats,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, '#8b5cf6'], [0.5, '#f59e0b'], [1, '#10b981']],
            line=dict(color='rgba(255, 255, 255, 0.15)', width=1)
        )
    ))

    fig.update_layout(
        **CYBER_THEME,
        title="Feature Importance Ranking (Gini Attribution)",
        xaxis_title="Importance Score",
        yaxis_title="Engineered Signal"
    )
    return fig
