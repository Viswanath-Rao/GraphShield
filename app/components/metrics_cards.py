"""
GraphShield UI Components - Next-Gen Metric & KPI Cards
Renders high-tech glassmorphism cards with glowing neon indicators.
"""

import streamlit as st
from typing import Optional

def render_metric_card(
    title: str,
    value: str,
    subtext: Optional[str] = None,
    badge_text: Optional[str] = None,
    badge_type: str = "low",  # low, medium, high, critical
    icon: str = "⚡"
):
    """Renders a futuristic cyber glass card with glowing status badge."""
    badge_html = ""
    if badge_text:
        badge_html = f'<div class="neon-badge badge-{badge_type}">{badge_text}</div>'
    
    subtext_html = ""
    if subtext:
        subtext_html = f'<div class="gs-metric-subtext">{subtext}</div>'

    html = f"""
    <div class="gs-card">
        <div class="gs-metric-label">
            <span>{icon}</span> {title}
        </div>
        <div class="gs-metric-val">{value}</div>
        {subtext_html}
        {badge_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
