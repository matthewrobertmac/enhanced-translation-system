# src/analysis/visualizations.py
import difflib
import re
import unicodedata
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from typing import List, Dict
from src.config import ENTITY_TYPES

# --- Optional Imports ---
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import networkx as nx
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- DIFF VIEWER ---
def create_diff_visualization(text1: str, text2: str, title1: str, title2: str) -> str:
    """
    Generate an HTML string showing a side-by-side diff of two texts.
    """
    words1 = text1.split()
    words2 = text2.split()
    
    d = difflib.Differ()
    diff = list(d.compare(words1, words2))
    
    # CSS helpers
    def _span(text, color, decoration="none", weight="normal"):
        return f"<span style='background:{color}; text-decoration:{decoration}; font-weight:{weight}; padding:2px 4px; border-radius:3px;'>{text}</span> "

    html = f"""
    <div style='display:flex; gap:20px; margin-bottom:20px;'>
        <div style='flex:1; background:#f8f9fa; padding:15px; border-radius:8px;'>
            <h4 style='color:#d9534f; margin-top:0;'>← {title1}</h4>
            <div style='background:white; padding:15px; border-left:4px solid #d9534f; line-height:1.6;'>
    """
    
    # Left side content (deletions)
    left_content = []
    for token in diff:
        word = token[2:]
        if token.startswith('- '): # Deleted
            left_content.append(_span(word, "#f8d7da", "line-through"))
        elif token.startswith('  '): # Unchanged
            left_content.append(f"<span style='color:#333'>{word}</span> ")
            
    html += "".join(left_content)
    html += """
            </div>
        </div>
        <div style='flex:1; background:#f8f9fa; padding:15px; border-radius:8px;'>
            <h4 style='color:#28a745; margin-top:0;'>{title2} →</h4>
            <div style='background:white; padding:15px; border-left:4px solid #28a745; line-height:1.6;'>
    """.format(title2=title2)
    
    # Right side content (insertions)
    right_content = []
    for token in diff:
        word = token[2:]
        if token.startswith('+ '): # Added
            right_content.append(_span(word, "#d4edda", "none", "bold"))
        elif token.startswith('  '): # Unchanged
            right_content.append(f"<span style='color:#333'>{word}</span> ")
            
    html += "".join(right_content)
    html += "</div></div></div>"
    
    return html

# --- WORD CLOUD ---
def compute_frequencies(text: str, stopwords: set) -> dict:
    """Compute word frequencies ignoring stopwords."""
    if not text: return {}
    # Normalize and clean
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s-]", " ", text).lower()
    
    freq = {}
    for w in text.split():
        w = w.strip("-")
        if len(w) > 1 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return freq

def render_wordcloud(freq_dict: dict, title: str = "Word Cloud"):
    """Render a wordcloud using Matplotlib."""
    if not WORDCLOUD_AVAILABLE:
        st.warning("WordCloud library not installed.")
        return
    if not freq_dict:
        st.info(f"No data for {title}")
        return

    try:
        wc = WordCloud(
            width=800, height=400, 
            background_color="white", 
            colormap="viridis"
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render word cloud: {str(e)}")

# --- ENTITY NETWORK ---
def render_entity_network(entities: List[Dict]):
    """Render an interactive NetworkX + Plotly graph."""
    if not PLOTLY_AVAILABLE or not entities:
        return

    G = nx.Graph()
    
    # Add Nodes
    for e in entities:
        G.add_node(e['name'], type=e['type'], count=e['count'])
    
    # Add Edges (Simple Clique Model: all entities in text are connected)
    # Limiting to top 30 entities to prevent lag
    top_entities = sorted(entities, key=lambda x: x['count'], reverse=True)[:30]
    names = [e['name'] for e in top_entities]
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            G.add_edge(name1, name2)
            
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Draw Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Metadata
        entity = next(e for e in entities if e['name'] == node)
        node_text.append(f"{node} ({entity['count']})<br>{entity['type']}")
        
        # Styling
        color = ENTITY_TYPES.get(entity['type'], {}).get('color', '#999')
        node_color.append(color)
        node_size.append(10 + min(entity['count'] * 2, 40))
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Entity Network (Top 30)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

def render_confidence_gauge(score: float):
    """Render a gauge chart for confidence score."""
    if not PLOTLY_AVAILABLE: return
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)