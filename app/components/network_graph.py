"""
GraphShield UI Components - High-Tech Network Graph Visualization
Renders glowing interactive force-directed topologies using NetworkX and Plotly.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np

def render_interactive_network(
    G: nx.DiGraph,
    title: str = "Financial Entity Transaction Graph",
    max_nodes: int = 150
) -> go.Figure:
    """
    Renders an interactive force-directed NetworkX graph with cyberpunk aesthetics.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No graph elements in buffer", showarrow=False, font=dict(color="#64748b"))
        return fig

    # Filter to top connected nodes if too large
    if G.number_of_nodes() > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        sub_G = G.subgraph(top_nodes).copy()
    else:
        sub_G = G

    # Spring layout coordinates
    pos = nx.spring_layout(sub_G, seed=42, k=0.38)

    edge_x = []
    edge_y = []
    for edge in sub_G.edges():
        u, v = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='rgba(255, 255, 255, 0.12)'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = sub_G.nodes[node]
        is_fraud = (data.get('risk') == 'Fraudulent' or data.get('fraud_count', 0) > 0)
        is_merchant = data.get('node_type') == 'Merchant'
        deg = sub_G.degree(node)

        # Size based on degree connectivity
        size = min(36, max(12, deg * 4))
        node_size.append(size)

        if is_fraud:
            node_color.append('#ef4444')  # Vivid Crimson
            status_str = "🚨 FRAUDULENT / MULE"
        elif is_merchant:
            node_color.append('#8b5cf6')  # Royal Violet
            status_str = "🏬 MERCHANT GATEWAY"
        elif deg > 5:
            node_color.append('#f59e0b')  # Amber High-Volume Hub
            status_str = "⚡ HIGH-VELOCITY HUB"
        else:
            node_color.append('#10b981')  # Emerald Mint
            status_str = "🛡️ VERIFIED CUSTOMER"

        hover_info = (
            f"<b>Node ID:</b> {node}<br>"
            f"<b>Role:</b> {status_str}<br>"
            f"<b>In-Degree:</b> {sub_G.in_degree(node)}<br>"
            f"<b>Out-Degree:</b> {sub_G.out_degree(node)}<br>"
            f"<b>Flagged Cascades:</b> {data.get('fraud_count', 0)}"
        )
        node_text.append(hover_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1.8, color='rgba(255, 255, 255, 0.85)')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(family='Plus Jakarta Sans', color='#ffffff', size=15)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=15, l=10, r=10, t=45),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='#090d1a',
            plot_bgcolor='#060913'
        )
    )

    return fig
